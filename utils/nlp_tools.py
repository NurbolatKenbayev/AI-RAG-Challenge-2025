import os
import sys

from sentence_transformers import SentenceTransformer


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)




def get_embeddings(text: str, embedding_model_path: str) -> list[float]:
    """
    Get embeddings for a given text using a specified embedding model.
    Args:
        text (str): The text to be embedded.
        embedding_model_path (str): The path to the embedding model.
    Returns:
        list[float]: The embeddings for the given text.
    """
    model = SentenceTransformer(embedding_model_path)
    return model.encode(text)