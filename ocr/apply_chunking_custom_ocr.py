import os
import sys

import warnings
warnings.filterwarnings("ignore")

from typing import Optional, List, Callable


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


try:
    from PIL import Image
except Exception:
    Image = None  # PDF composition from images will be disabled if Pillow is missing


from dotenv import load_dotenv
load_dotenv()


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.docs_parser import (
    converter,
    chunker,
    custom_tokenizer,
    save_doc_as_json,
    print_section_headers,
    print_extracted_chunks,
    parse_pdf,
)

from docling_core.transforms.chunker.hierarchical_chunker import DocChunk

from .interfaces import OCRModule, OCRResult
from .hooks import ensure_hook_list, DocPreprocessHook, DoclingDocHook, ChunksHook


def _compose_pdf_from_images(image_paths: List[str], output_pdf_path: str) -> Optional[str]:
    """
    Compose a multi-page PDF from a list of image paths.
    Returns the created PDF path, or None on failure.
    """
    if Image is None:
        logger.warning("Pillow is not available; cannot compose PDF from images.")
        return None

    if not image_paths:
        logger.warning("No image paths provided for PDF composition.")
        return None

    try:
        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

        first_image = Image.open(image_paths[0]).convert("RGB")
        additional_pages = []
        for p in image_paths[1:]:
            additional_pages.append(Image.open(p).convert("RGB"))

        first_image.save(output_pdf_path, save_all=True, append_images=additional_pages)
        return output_pdf_path
    except Exception as e:
        logger.error(f"Failed to compose PDF from images: {e}")
        return None


def _resolve_ocr_input_path(
    original_path: str,
    ocr_result: OCRResult,
) -> str:
    """
    Prefer a searchable PDF produced by OCR; if only images are present, compose a PDF; otherwise fallback to original.
    """
    if ocr_result.output_pdf_path and os.path.exists(ocr_result.output_pdf_path):
        return ocr_result.output_pdf_path

    if ocr_result.page_image_paths:
        tmp_dir = os.path.join(os.path.dirname(original_path), "ocr_tmp")
        composed_pdf = os.path.join(tmp_dir, os.path.basename(original_path))
        pdf_path = _compose_pdf_from_images(ocr_result.page_image_paths, composed_pdf)
        if pdf_path and os.path.exists(pdf_path):
            return pdf_path

    logger.warning("OCR did not produce a PDF or images; falling back to the original file path.")
    return original_path


def apply_chunking_custom_ocr(
    pdf_path: str,
    ocr_module: Optional[OCRModule] = None,
    pre_hooks: Optional[List[DocPreprocessHook]] = None,
    doc_hook: Optional[DoclingDocHook] = None,
    chunks_hook: Optional[ChunksHook] = None,
    save_doc_dir: Optional[str] = None,
    print_titles: bool = False,
    save_chunks_dir: Optional[str] = None,
) -> List[DocChunk]:
    """
    A pluggable variant of utils.docs_parser.apply_chunking with optional custom OCR and hooks.

    - If `ocr_module` is provided, it is used to preprocess the input into a searchable PDF (preferred)
      or page images (fallback), which are then parsed by Docling.
    - If not provided, behaves like the original apply_chunking using the given `pdf_path`.
    - Optional hooks allow pre-processing the raw input path, mutating the Docling document, and
      post-processing the resulting chunks.
    """
    # 1) Run pre-processing hooks on the input path (e.g., deskew, denoise)
    current_input_path = pdf_path
    for hook in ensure_hook_list(pre_hooks):
        try:
            current_input_path = hook(current_input_path)
        except Exception as e:
            logger.warning(f"Preprocess hook failed; continuing with previous path. Error: {e}")

    # 2) Apply OCR module if provided
    ocr_result: Optional[OCRResult] = None
    if ocr_module is not None:
        try:
            ocr_result = ocr_module.preprocess(current_input_path)
            current_input_path = _resolve_ocr_input_path(original_path=current_input_path, ocr_result=ocr_result)
        except Exception as e:
            logger.warning(f"OCR module failed; continuing with original file. Error: {e}")
            current_input_path = pdf_path

    # 3) Parse with Docling
    logger.info("Semantic segmentation for given document (parsing)...")
    doc = parse_pdf(converter=converter, pdf_path=current_input_path)

    # Optional save DoclingDocument as JSON
    if save_doc_dir:
        try:
            os.makedirs(save_doc_dir, exist_ok=True)
            json_path = os.path.join(save_doc_dir, os.path.basename(pdf_path).split(".")[0] + "_doc.json")
            save_doc_as_json(doc=doc, json_path=json_path)
        except Exception as e:
            logger.warning(f"Failed to save DoclingDocument JSON: {e}")

    # Optional print titles
    if print_titles:
        logger.info("Extracted section headers:\n")
        try:
            print_section_headers(doc=doc)
        except Exception as e:
            logger.warning(f"Failed to print section headers: {e}")

    # 4) Optional doc-level hook before chunking
    if doc_hook is not None:
        try:
            doc = doc_hook(doc)
        except Exception as e:
            logger.warning(f"Doc hook failed; proceeding with unmodified document. Error: {e}")

    # 5) Chunking
    logger.info("Chunking...")
    try:
        chunk_iter = chunker.chunk(dl_doc=doc)
        chunks: List[DocChunk] = list(chunk_iter)
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        chunks = []

    # 6) Optional chunks hook
    if chunks_hook is not None:
        try:
            chunks = chunks_hook(chunks)
        except Exception as e:
            logger.warning(f"Chunks hook failed; returning original chunks. Error: {e}")

    # 7) Optional save extracted chunks
    if save_chunks_dir:
        try:
            os.makedirs(save_chunks_dir, exist_ok=True)
            save_path = os.path.join(save_chunks_dir, os.path.basename(pdf_path).split(".")[0] + "_chunks.txt")
            print_extracted_chunks(chunks=chunks, tokenizer=custom_tokenizer, chunker=chunker, save_path=save_path)
        except Exception as e:
            logger.warning(f"Failed to save extracted chunks: {e}")

    # 8) OCR cleanup
    if ocr_module is not None and ocr_result is not None:
        try:
            ocr_module.cleanup(ocr_result)
        except Exception as e:
            logger.warning(f"OCR cleanup failed: {e}")

    return chunks


