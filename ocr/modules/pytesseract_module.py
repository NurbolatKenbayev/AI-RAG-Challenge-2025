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


from ..interfaces import OCRModule, OCRResult


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _render_pdf_to_images(pdf_path: str, dpi: int, out_dir: str) -> List[str]:
    """
    Render a PDF into per-page PNG images using pypdfium2.
    Returns a list of image file paths.
    """
    import pypdfium2 as pdfium

    _ensure_dir(out_dir)
    images: List[str] = []

    pdf = pdfium.PdfDocument(pdf_path)
    scale = dpi / 72.0
    for i in range(len(pdf)):
        page = pdf[i]
        pil_image = page.render(scale=scale).to_pil()
        img_path = os.path.join(out_dir, f"page_{i+1:04d}.png")
        pil_image.save(img_path, format="PNG")
        images.append(img_path)

    return images


def _merge_pdfs(page_pdf_paths: List[str], output_pdf_path: str) -> Optional[str]:
    """
    Merge per-page PDFs into a single PDF if PyPDF2 is available.
    Returns the output path on success, otherwise None.
    """
    try:
        from PyPDF2 import PdfMerger
    except Exception:
        logger.warning("PyPDF2 not available; will not merge per-page OCR PDFs.")
        return None

    try:
        merger = PdfMerger()
        for p in page_pdf_paths:
            merger.append(p)
        _ensure_dir(os.path.dirname(output_pdf_path))
        with open(output_pdf_path, "wb") as f:
            merger.write(f)
        merger.close()
        return output_pdf_path
    except Exception as e:
        logger.error(f"Failed to merge PDFs: {e}")
        return None


class PytesseractOCRModule:
    """
    OCR module that uses pytesseract with Russian language.

    Behavior:
    - For PDF input: renders pages to images, runs OCR per page to create searchable PDFs,
      tries to merge them into a single PDF. If merging is unavailable, returns per-page images
      to allow a fallback PDF composition without text layer.
    - For image input: runs OCR directly and returns a single-page PDF.

    Configuration via env vars:
    - TESSERACT_CMD: full path to tesseract binary (especially on Windows)
    - PYTESS_LANG: override language, default 'rus'
    - PYTESS_DPI: rendering DPI for PDF pages, default 300
    """

    def __init__(self, language: Optional[str] = None, dpi: Optional[int] = None):
        import pytesseract

        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Enforce Russian OCR by default; ignore environment overrides
        self.language = language or "rus"
        try:
            self.dpi = int(dpi or os.getenv("PYTESS_DPI", "300"))
        except Exception:
            self.dpi = 300

        self._pytesseract = pytesseract

    def preprocess(self, input_path: str) -> OCRResult:
        base_dir = os.path.dirname(os.path.abspath(input_path))
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        tmp_root = os.path.join(base_dir, "ocr_pytesseract_tmp", base_name)
        img_dir = os.path.join(tmp_root, "images")
        page_pdf_dir = os.path.join(tmp_root, "page_pdfs")

        _ensure_dir(img_dir)
        _ensure_dir(page_pdf_dir)

        ext = os.path.splitext(input_path)[1].lower()

        page_image_paths: List[str] = []
        page_pdf_paths: List[str] = []

        if ext == ".pdf":
            # Render PDF pages to images
            page_image_paths = _render_pdf_to_images(pdf_path=input_path, dpi=self.dpi, out_dir=img_dir)
        else:
            # Assume it's an image-compatible format
            page_image_paths = [input_path]

        # Run OCR on each page image to produce a searchable PDF page
        for idx, img_path in enumerate(page_image_paths, start=1):
            try:
                pdf_bytes = self._pytesseract.image_to_pdf_or_hocr(img_path, lang=self.language, extension='pdf')
                out_page_pdf = os.path.join(page_pdf_dir, f"page_{idx:04d}.pdf")
                with open(out_page_pdf, "wb") as f:
                    f.write(pdf_bytes)
                page_pdf_paths.append(out_page_pdf)
            except Exception as e:
                logger.warning(f"pytesseract OCR failed for {img_path}: {e}")

        output_pdf_path: Optional[str] = None
        if len(page_pdf_paths) == 1:
            # Single page: use as final
            output_pdf_path = os.path.join(tmp_root, f"{base_name}_ocr.pdf")
            try:
                with open(page_pdf_paths[0], "rb") as src, open(output_pdf_path, "wb") as dst:
                    dst.write(src.read())
            except Exception as e:
                logger.warning(f"Failed to copy single-page OCR PDF: {e}")
                output_pdf_path = None
        elif len(page_pdf_paths) > 1:
            # Merge multi-page into a final PDF
            output_pdf_path = _merge_pdfs(page_pdf_paths, os.path.join(tmp_root, f"{base_name}_ocr.pdf"))

        return OCRResult(
            output_pdf_path=output_pdf_path,
            page_image_paths=page_image_paths,
            metadata={
                "tmp_root": tmp_root,
                "num_pages": len(page_image_paths),
                "language": self.language,
                "dpi": self.dpi,
                "merged": bool(output_pdf_path),
            },
        )

    def cleanup(self, result: OCRResult) -> None:
        # By default, keep artifacts for debugging; implementers may choose to delete tmp dirs.
        # Example cleanup code (commented out):
        # import shutil
        # tmp_root = (result.metadata or {}).get("tmp_root")
        # if isinstance(tmp_root, str) and os.path.exists(tmp_root):
        #     try:
        #         shutil.rmtree(tmp_root)
        #     except Exception:
        #         pass
        return


