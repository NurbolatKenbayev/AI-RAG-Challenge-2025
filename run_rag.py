import os
import sys

import warnings
warnings.filterwarnings("ignore")


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_modules.retriever import Retriever
from core_modules.generator import Generator


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

WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "public_dataset")




def rag_query(user_query: str, retriever: Retriever, generator: Generator, top_k: int = 5):
    collection = retriever.get_collection(WEAVIATE_COLLECTION_NAME)
    if collection is None:
        logger.error(f"Collection {WEAVIATE_COLLECTION_NAME} not found")
        return None

    weaviate_objects = retriever.hybrid_search(
        query_text=user_query, 
        collection=collection,
        top_k=top_k
    )

    generated_answer = generator.generate(
        user_query=user_query, 
        retrieved_context=weaviate_objects
    )
    
    return generated_answer


if __name__ == "__main__":
    user_query = "9 npoyue Hanoru K ynnare?"

    retriever = Retriever()
    generator = Generator()
    
    generated_answer = rag_query(user_query, retriever, generator, top_k=3)
    print(generated_answer)

    # Close Weaviate client
    retriever.close_client()
