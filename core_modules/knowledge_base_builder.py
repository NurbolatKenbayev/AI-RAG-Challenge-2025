import os
import sys

import warnings
warnings.filterwarnings("ignore")

import asyncio
import uuid
import glob

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.docs_parser import custom_tokenizer, chunker, apply_chunking
from ocr.apply_chunking_custom_ocr import apply_chunking_custom_ocr
from ocr.registry import get_ocr_module
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


weaviate_client = weaviate.connect_to_custom(
    http_host=WEAVIATE_HTTP_HOST,     # HTTP hostname or IP
    http_port=WEAVIATE_HTTP_PORT,     # HTTP port
    http_secure=False,                # True for HTTPS
    grpc_host=WEAVIATE_GRPC_HOST,     # gRPC hostname (usually same as http_host) 
    grpc_port=WEAVIATE_GRPC_PORT,     # gRPC port (default: 50051)
    grpc_secure=False,                # True for secure gRPC
    auth_credentials=Auth.api_key(WEAVIATE_AUTH_KEY)
)




async def check_services_availability(weaviate_client: weaviate.client.WeaviateClient):
    if not weaviate_client.is_ready():
        logger.error("Weaviate service is not available.")
        return False
    return True


async def create_weaviate_collection(weaviate_client: weaviate.client.WeaviateClient, collection_name: str):
    try:
        if weaviate_client.collections.exists(collection_name):
            logger.info(f"Collection {collection_name} already exists.")
            return
        
        logger.info(f"Creating collection {collection_name}.")
        # Define the schema for the DocumentChunk class
        weaviate_client.collections.create(
            name=collection_name,
            vector_config=Configure.Vectors.self_provided(),  # we'll provide our own vectors
            properties=[
                Property(name="content", data_type=DataType.TEXT, index_searchable=True, index_filterable=True),
                Property(name="filename", data_type=DataType.TEXT, index_searchable=True, index_filterable=True),
                Property(name="chunk_id", data_type=DataType.UUID, index_searchable=False, index_filterable=False),
                Property(name="chunk_index", data_type=DataType.INT, index_searchable=False, index_filterable=True, index_range_filters=True),
                Property(name="num_tokens", data_type=DataType.INT, index_searchable=False, index_filterable=True, index_range_filters=True),
                Property(name="headings", data_type=DataType.TEXT_ARRAY, index_searchable=True, index_filterable=True),
                Property(name="first_element_page_number", data_type=DataType.INT, index_searchable=False, index_filterable=True, index_range_filters=True),
                Property(name="last_element_page_number", data_type=DataType.INT, index_searchable=False, index_filterable=True, index_range_filters=True),
                Property(name="doc_items_refs", data_type=DataType.TEXT_ARRAY, index_searchable=False, index_filterable=True),
                # add more metadata properties as needed, e.g. author, title, etc.
            ]
        )
    except Exception as e:
        logger.error(f"Error creating Weaviate collection: {e}")
        raise


async def process_document(document_path: str, weaviate_client: weaviate.client.WeaviateClient, collection_name: str):
    ### Step 1: File parsing and chunking
    logger.info(f"Applying chunking to: {document_path}")
    document_dir = os.path.dirname(document_path)
    # chunks = apply_chunking(
    #     pdf_path=document_path,
    #     save_doc_dir=os.path.join(document_dir, "docling", "doc"),
    #     print_titles=False, 
    #     save_chunks_dir=os.path.join(document_dir, "docling", "chunks")
    # )
    chunks = apply_chunking_custom_ocr(
        pdf_path=document_path,
        ocr_module=get_ocr_module("pytesseract"),  # or None to skip OCR
        pre_hooks=None,
        doc_hook=None,
        chunks_hook=None,
        save_doc_dir=os.path.join(document_dir, "docling", "doc"),
        print_titles=False,
        save_chunks_dir=os.path.join(document_dir, "docling", "chunks"),
    )
    
    ### Step 2: Embedding and Indexing
    logger.info(f"Embedding {len(chunks)} chunks and uploading to Weaviate.")
    
    # Get Weaviate collection
    weaviate_collection = weaviate_client.collections.get(collection_name)
    initial_count = weaviate_collection.aggregate.over_all(total_count=True).total_count

    # Using batch.fixed_size for controlled batch processing
    with weaviate_collection.batch.fixed_size(batch_size=100) as batch:
        for idx, chunk in enumerate(chunks):

            chunk_text_with_headings = chunker.contextualize(chunk=chunk)

            chunk_embedding = get_embeddings(
                text=chunk_text_with_headings,
                embedding_model_path=EMBEDDING_MODEL_PATH
            )

            # Generate unique ID for both systems
            chunk_id = str(uuid.uuid4())
            
            # Create object for Weaviate
            batch.add_object(
                properties={
                    "content": chunk_text_with_headings,
                    "filename": os.path.basename(document_path),
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "num_tokens": custom_tokenizer.count_tokens(text=chunk_text_with_headings),
                    "headings": chunk.meta.headings,
                    "first_element_page_number": int(chunk.meta.doc_items[0].prov[0].page_no),
                    "last_element_page_number": int(chunk.meta.doc_items[-1].prov[0].page_no),
                    "doc_items_refs": [item.self_ref for item in chunk.meta.doc_items],
                },
                vector=chunk_embedding
            )
    
    if weaviate_collection.aggregate.over_all(total_count=True).total_count - initial_count == len(chunks):
        logger.info(f"Uploaded {len(chunks)} chunks from {document_path} to Weaviate.")
        return True
    else:
        logger.error(f"Failed to upload {len(chunks)} chunks from {document_path} to Weaviate.")
        return False


async def main(document_path_list: list[str]):
    logger.info(f"Found {len(document_path_list)} files")
    logger.info("{files}".format(files="\n".join([os.path.basename(file) for file in document_path_list])))

    if not await check_services_availability(weaviate_client=weaviate_client):
        logger.error("Not all services are available. Please fix that and start again.")
        return
    else:
        logger.info("All services are available. Starting the process.")
        logger.info("-"*40)

    await create_weaviate_collection(weaviate_client=weaviate_client, collection_name=WEAVIATE_COLLECTION_NAME)

    success_count = 0
    failed_count = 0
    failed_document_path_list = []
    for document_path in document_path_list:
        success = await process_document(
            document_path=document_path, 
            weaviate_client=weaviate_client, 
            collection_name=WEAVIATE_COLLECTION_NAME
        )
        if success:
            logger.info(f"Successfully processed {document_path}")
            success_count += 1
        else:
            logger.error(f"Failed to process {document_path}")
            failed_count += 1
            failed_document_path_list.append(document_path)

    logger.info(f"Successfully processed {success_count}/{len(document_path_list)} documents.")
    logger.info(f"Failed to process {failed_count}/{len(document_path_list)} documents.")
    logger.info(f"Failed document paths: {failed_document_path_list}")


def get_files_from_dir(dir_path: str, file_type: str = "pdf"):
    dir_companies = glob.glob(dir_path+"/*")

    document_path_list = []
    for company in dir_companies:
        selected_files = [item for item in os.listdir(company) if item.endswith(f".{file_type}")]
        for pdf_file in selected_files:
            file_path = os.path.join(company, pdf_file)
            document_path_list.append(file_path)
    return document_path_list


if __name__ == "__main__":
    # Pass your list of PDF file paths here
    # document_path_list = [
    #     # os.path.expanduser("~/Desktop/work_dir/68a58892b4058539485342/Dataset/AsiaAgroFood/aafdf5_2024_cons_rus.pdf"),
    #     # "/Users/nurbolatkenbayev/Desktop/work_dir/68a58892b4058539485342/Dataset/Rakhat/rahtp_2024_rus.pdf"
    #     # "/Users/nurbolatkenbayev/Desktop/work_dir/68a58892b4058539485342/Dataset/AsiaAgroFood/aafd_af_4_2025.pdf"
    #     # "/Users/nurbolatkenbayev/Desktop/work_dir/68a58892b4058539485342/Dataset/BRK/brkz_af_4_2025.pdf"
    # ]
    
    data_dir = os.path.expanduser("~/Desktop/work_dir/68a58892b4058539485342/Dataset")
    file_type = "pdf"
    # file_type = "xlsx"

    document_path_list = get_files_from_dir(dir_path=data_dir, file_type=file_type)

    asyncio.run(main(document_path_list))

    weaviate_client.close()