import os
import sys

import warnings
warnings.filterwarnings("ignore")

from typing import Protocol, Callable, List, Optional


class DocPreprocessHook(Protocol):
    """
    Pre-processing hook applied to the raw input path before OCR.
    Should return a path to be used as OCR input (can be the same path).
    """
    def __call__(self, input_path: str) -> str: ...


class DoclingDocHook(Protocol):
    """
    Hook to mutate or filter a DoclingDocument after conversion and before chunking.
    Note: keep import-free to avoid tight coupling; the orchestrator will type-ignore.
    """
    def __call__(self, docling_document): ...


class ChunksHook(Protocol):
    """
    Hook to post-process the list of chunks before returning.
    """
    def __call__(self, chunks): ...


# Utility to normalize hook lists
def ensure_hook_list(hooks: Optional[List[Callable]]) -> List[Callable]:
    return list(hooks) if hooks else []


