from .interfaces import OCRModule, OCRResult
from .hooks import DocPreprocessHook, DoclingDocHook, ChunksHook
from .apply_chunking_custom_ocr import apply_chunking_custom_ocr
from .registry import get_ocr_module, register_ocr_module, build_from_env

__all__ = [
    "OCRModule",
    "OCRResult",
    "DocPreprocessHook",
    "DoclingDocHook",
    "ChunksHook",
    "apply_chunking_custom_ocr",
    "get_ocr_module",
    "register_ocr_module",
    "build_from_env",
]


