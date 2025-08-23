import os
import sys

import warnings
warnings.filterwarnings("ignore")

from typing import Callable, Optional, Dict

from .interfaces import OCRModule


_REGISTRY: Dict[str, Callable[[], OCRModule]] = {}


def register_ocr_module(name: str, factory: Callable[[], OCRModule]) -> None:
    """Register an OCR module factory under a name."""
    _REGISTRY[name] = factory


def _lazy_example_factory() -> OCRModule:
    from .modules.example_module import ExampleOCRModule
    return ExampleOCRModule()


# Pre-register an example module for convenience
register_ocr_module("example", _lazy_example_factory)


def _lazy_pytesseract_factory() -> OCRModule:
    from .modules.pytesseract_module import PytesseractOCRModule
    return PytesseractOCRModule()


# Pre-register pytesseract module
register_ocr_module("pytesseract", _lazy_pytesseract_factory)


def _lazy_paligemma_factory() -> OCRModule:
    from .modules.paligemma_module import PaligemmaOCRModule
    return PaligemmaOCRModule()


# Pre-register paligemma module
register_ocr_module("paligemma", _lazy_paligemma_factory)


def get_ocr_module(name: Optional[str]) -> Optional[OCRModule]:
    """
    Resolve an OCR module by name.

    - If name is None or empty, returns None (no OCR, default Docling path).
    - If name is in the registry, instantiates via its factory.
    - If name looks like "package.module:ClassName", attempts dynamic import and instantiation.
    """
    if not name:
        return None

    if name in _REGISTRY:
        return _REGISTRY[name]()

    # Dynamic import path support: "pkg.mod:Class"
    if ":" in name:
        module_path, class_name = name.split(":", 1)
        try:
            import importlib
            module = importlib.import_module(module_path)
            klass = getattr(module, class_name)
            return klass()  # type: ignore[call-arg]
        except Exception:
            return None

    return None


def build_from_env(env_var: str = "CUSTOM_OCR_NAME") -> Optional[OCRModule]:
    """
    Convenience helper to construct an OCR module from an environment variable.
    """
    name = os.getenv(env_var)
    return get_ocr_module(name)


