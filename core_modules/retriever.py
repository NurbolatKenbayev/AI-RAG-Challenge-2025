import os
import sys

import warnings
warnings.filterwarnings("ignore")

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nlp_tools import get_embeddings


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

WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "public_dataset")
WEAVIATE_HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
WEAVIATE_HTTP_PORT = os.getenv("WEAVIATE_HTTP_PORT", "8080")
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "localhost")
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT", "50051")
WEAVIATE_AUTH_KEY = os.getenv("WEAVIATE_AUTH_KEY", None)




class Retriever:
    def __init__(self):
        self.weaviate_client = weaviate.connect_to_custom(
            http_host=WEAVIATE_HTTP_HOST,     # HTTP hostname or IP
            http_port=WEAVIATE_HTTP_PORT,     # HTTP port
            http_secure=False,                # True for HTTPS
            grpc_host=WEAVIATE_GRPC_HOST,     # gRPC hostname (usually same as http_host) 
            grpc_port=WEAVIATE_GRPC_PORT,     # gRPC port (default: 50051)
            grpc_secure=False,                # True for secure gRPC
            auth_credentials=Auth.api_key(WEAVIATE_AUTH_KEY)
        )


    def close_client(self):
        self.weaviate_client.close()


    def get_collection(self, collection_name: str) -> weaviate.collections.collection.sync.Collection:
        try:
            collection = self.weaviate_client.collections.get(collection_name)
            return collection
        except Exception as e:
            logger.error(f"Error getting collection {collection_name}: {e}")
            return None


    def hybrid_search(self, query_text: str, collection: weaviate.collections.collection.sync.Collection, top_k: int = 5) -> list[weaviate.collections.classes.internal.Object]:    
        logger.info("Generating embedding for query...")
        query_vector = get_embeddings(
            text=query_text,
            embedding_model_path=EMBEDDING_MODEL_PATH
        )
        query_vector_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        logger.info(f"Query vector shape: {len(query_vector_list)}")

        logger.info("Performing hybrid search...")
        response = collection.query.hybrid(
            query=query_text,
            vector=query_vector_list,
            alpha=0.5, # 0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced
            limit=top_k,
            return_metadata=MetadataQuery(score=True, explain_score=True)
        )

        logger.debug(f"Hybrid search results for: '{query_text}'\n")
        for i, obj in enumerate(response.objects, 1):
            score = obj.metadata.score if hasattr(obj.metadata, 'score') else "N/A"
            explain_score = obj.metadata.explain_score if hasattr(obj.metadata, 'explain_score') else "N/A"
            logger.debug(f"Result {i}:")
            logger.debug(f"  Score: {score:.4f}" if isinstance(score, (int, float)) else f"  Score: {score}")
            logger.debug(f"  Explain Score: {explain_score}")
            logger.debug(f"  chunk_id: {obj.properties['chunk_id']}")
            logger.debug(f"  filename: {obj.properties['filename']}")
            logger.debug(f"  first_element_page_number: {obj.properties['first_element_page_number']}")
            logger.debug(f"  content with headings: {obj.properties['content']}")
            logger.debug("\n")

        return response.objects

    
if __name__ == "__main__":
    query_text = "9 npoyue Hanoru K ynnare?"

    retriever = Retriever()

    collection = retriever.get_collection(WEAVIATE_COLLECTION_NAME)
    if collection is None:
        logger.error(f"Collection {WEAVIATE_COLLECTION_NAME} not found")

    weaviate_objects = retriever.hybrid_search(query_text, collection)

    retriever.close_client()