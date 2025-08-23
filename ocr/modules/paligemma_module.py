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


class PaligemmaOCRModule:
    """
    OCR module that uses Google's PaliGemma VLM to transcribe page images into Russian text.

    Note: This module extracts text but does not embed a searchable text layer into the PDF.
    It returns `page_image_paths` for downstream PDF composition and saves recognized text
    files alongside for potential custom processing hooks.

    Configuration via env vars:
    - PALIGEMMA_MODEL: HF model id (default: 'google/paligemma-3b-mix-224')
    - PALIGEMMA_MAX_TOKENS: max new tokens to generate per page (default: 1024)
    - PALIGEMMA_DPI: rendering DPI for PDF pages, default 200
    """

    def __init__(self, model_name: Optional[str] = None, dpi: Optional[int] = None, max_new_tokens: Optional[int] = None):
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self.model_name = model_name or os.getenv("PALIGEMMA_MODEL", "google/paligemma-3b-mix-224")
        try:
            self.dpi = int(dpi or os.getenv("PALIGEMMA_DPI", "200"))
        except Exception:
            self.dpi = 200

        try:
            self.max_new_tokens = int(max_new_tokens or os.getenv("PALIGEMMA_MAX_TOKENS", "1024"))
        except Exception:
            self.max_new_tokens = 1024

        device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else (torch.bfloat16 if device == "cpu" else torch.float16)

        logger.info(f"Loading Paligemma model: {self.model_name} on {device}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=dtype, device_map="auto")
        self.device = device

    def _transcribe_image_ru(self, image_path: str) -> str:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        # Russian transcription prompt
        prompt = "Распознай весь текст на изображении и выведи его без каких-либо объяснений."

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        generated = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Model often echoes the prompt; try to strip it
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        return generated.strip()

    def preprocess(self, input_path: str) -> OCRResult:
        base_dir = os.path.dirname(os.path.abspath(input_path))
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        tmp_root = os.path.join(base_dir, "ocr_paligemma_tmp", base_name)
        img_dir = os.path.join(tmp_root, "images")
        text_dir = os.path.join(tmp_root, "page_texts")
        _ensure_dir(img_dir)
        _ensure_dir(text_dir)

        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".pdf":
            page_image_paths = _render_pdf_to_images(pdf_path=input_path, dpi=self.dpi, out_dir=img_dir)
        else:
            page_image_paths = [input_path]

        recognized_texts: List[str] = []
        for idx, img_path in enumerate(page_image_paths, start=1):
            try:
                text = self._transcribe_image_ru(img_path)
                recognized_texts.append(text)
                with open(os.path.join(text_dir, f"page_{idx:04d}.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                logger.warning(f"Paligemma OCR failed for {img_path}: {e}")
                recognized_texts.append("")

        # We are not creating a searchable PDF layer here; return images for downstream composition
        return OCRResult(
            output_pdf_path=None,
            page_image_paths=page_image_paths,
            metadata={
                "tmp_root": tmp_root,
                "num_pages": len(page_image_paths),
                "recognized_texts": recognized_texts,
                "model": self.model_name,
                "dpi": self.dpi,
            },
        )

    def cleanup(self, result: OCRResult) -> None:
        # Keep artifacts by default for inspection
        return


