import os
import sys

import warnings
warnings.filterwarnings("ignore")

from typing import Optional

from ..interfaces import OCRModule, OCRResult


class ExampleOCRModule:
    """
    A minimal example OCR module that performs no transformation.

    - Returns no output_pdf_path or images; the pipeline will fallback to the original input.
    - Serves as a template for implementing real OCR integrations.
    """

    def preprocess(self, input_path: str) -> OCRResult:
        return OCRResult(
            output_pdf_path=None,
            page_image_paths=None,
            metadata={"note": "ExampleOCRModule did not alter the input; fallback will be used."},
        )

    def cleanup(self, result: OCRResult) -> None:
        # No temporary artifacts to delete in the example module
        return


