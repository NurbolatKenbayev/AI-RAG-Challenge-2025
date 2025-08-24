import os
import sys

import warnings
warnings.filterwarnings("ignore")

from typing import Optional, List

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


from dotenv import load_dotenv
load_dotenv()


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.docs_parser import (
    converter,
    chunker,
    custom_tokenizer,
    save_doc_as_json,
    print_section_headers,
)

from docling_core.transforms.chunker.hierarchical_chunker import DocChunk


def _parse_excel(excel_path: str):
    """
    Parse an Excel document using the shared DocumentConverter.
    This leverages the same converter pipeline used for PDFs; if the
    backend supports spreadsheets, this will return a DoclingDocument.
    """
    # The shared converter autodetects by file extension / content
    doc = converter.convert(source=excel_path).document
    return doc


def apply_chunking_excel(
    excel_path: str,
    save_doc_dir: Optional[str] = None,
    print_titles: bool = False,
    save_chunks_dir: Optional[str] = None,
) -> List[DocChunk]:
    """
    Apply the same chunking flow as utils.docs_parser.apply_chunking, but for Excel files.
    The function parses the Excel into a DoclingDocument, optionally saves the JSON
    representation, optionally prints section headers, performs chunking, and optionally
    saves extracted chunks to disk.
    """

    logger.info(f"Semantic segmentation for given Excel document (parsing)...")
    doc = _parse_excel(excel_path=excel_path)

    if save_doc_dir:
        os.makedirs(save_doc_dir, exist_ok=True)
        json_path = os.path.join(save_doc_dir, os.path.basename(excel_path).split(".")[0] + "_doc.json")
        save_doc_as_json(doc=doc, json_path=json_path)

    if print_titles:
        logger.info("Extracted section headers:\n")
        print_section_headers(doc=doc)

    logger.info("Chunking...")
    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks: List[DocChunk] = list(chunk_iter)

    if save_chunks_dir:
        os.makedirs(save_chunks_dir, exist_ok=True)
        save_path = os.path.join(save_chunks_dir, os.path.basename(excel_path).split(".")[0] + "_chunks.txt")
        from utils.docs_parser import print_extracted_chunks  # local import to avoid circulars
        print_extracted_chunks(chunks=chunks, tokenizer=custom_tokenizer, chunker=chunker, save_path=save_path)

    return chunks


