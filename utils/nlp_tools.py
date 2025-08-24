import os
import sys

from sentence_transformers import SentenceTransformer
import torch




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

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "deepvk/USER2-base")
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "jeffwan/mmarco-mMiniLMv2-L12-H384-v1")


model = SentenceTransformer(EMBEDDING_MODEL_PATH)


def get_embeddings(text: str, embedding_model_path: str) -> list[float]:
    """
    Get embeddings for a given text using a specified embedding model.
    Args:
        text (str): The text to be embedded.
        embedding_model_path (str): The path to the embedding model.
    Returns:
        list[float]: The embeddings for the given text.
    """
    # Force eager attention to avoid PyTorch compatibility issues with ModernBERT
    try:
        # Use newer API if available  
        from torch.nn.attention import sdpa_kernel
        context_manager = sdpa_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    except (ImportError, TypeError):
        # Fallback to older API or handle different parameter names
        context_manager = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    
    with context_manager:
        # Force the model to use eager attention implementation
        if hasattr(model[0].auto_model, 'config') and hasattr(model[0].auto_model.config, '_attn_implementation'):
            model[0].auto_model.config._attn_implementation = "eager"
        return model.encode(text)


def get_embeddings_batch(texts: list[str], embedding_model_path: str, batch_size: int = 128) -> list[float]:
    """
    Get embeddings for a given text using a specified embedding model.
    Args:
        text (str): The text to be embedded.
        embedding_model_path (str): The path to the embedding model.
    Returns:
        list[float]: The embeddings for the given text.
    """
    # Force eager attention to avoid PyTorch compatibility issues with ModernBERT
    try:
        # Use newer API if available  
        from torch.nn.attention import sdpa_kernel
        context_manager = sdpa_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    except (ImportError, TypeError):
        # Fallback to older API or handle different parameter names
        context_manager = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    
    with context_manager:
        # Force the model to use eager attention implementation
        if hasattr(model[0].auto_model, 'config') and hasattr(model[0].auto_model.config, '_attn_implementation'):
            model[0].auto_model.config._attn_implementation = "eager"
        return model.encode(texts, batch_size=batch_size, show_progress_bar=True)