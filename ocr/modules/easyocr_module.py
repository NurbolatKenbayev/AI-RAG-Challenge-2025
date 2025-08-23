import os
import sys

import warnings
warnings.filterwarnings("ignore")

from typing import Optional

from ..interfaces import OCRModule, OCRResult


class EasyOCROCRModule:
    """
    Skeleton for an EasyOCR-backed OCR module.

    This is a template and does NOT yet generate a searchable PDF. Implementers can:
    - Use EasyOCR to extract text per page.
    - Optionally render text onto a PDF (e.g., via reportlab) or produce per-page images and
      let the orchestrator compose a PDF.
    - Return OCRResult with either `output_pdf_path` or `page_image_paths` populated.
    """

    def __init__(self, *args, **kwargs):
        pass

    def preprocess(self, input_path: str) -> OCRResult:
        # Placeholder implementation; fall back to the original input
        return OCRResult(
            output_pdf_path=None,
            page_image_paths=None,
            metadata={"note": "EasyOCROCRModule template; no OCR performed."},
        )

    def cleanup(self, result: OCRResult) -> None:
        return


