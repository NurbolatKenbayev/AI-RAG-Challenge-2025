import os
import sys

import warnings
warnings.filterwarnings("ignore")

from docling.document_converter import DocumentConverter

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    DocChunk,
)
from docling_core.transforms.serializer.base import BaseDocSerializer, BaseTableSerializer, SerializationResult
from docling_core.transforms.serializer.html import HTMLDocSerializer
from docling_core.types.doc.document import TableItem, DoclingDocument


from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from docling_core.types.doc.labels import DocItemLabel


import time
import json
from contextlib import redirect_stdout


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

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")




class HTMLTableSerializer(BaseTableSerializer):
    """HTML-based table item serializer."""
    
    def serialize(
        self,
        *,
        item: TableItem,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the table item to HTML format."""
        # Create an HTML doc serializer for this specific table
        html_serializer = HTMLDocSerializer(doc=doc)
        
        # Use the HTML serializer to serialize the table
        result = html_serializer.serialize(item=item)
        
        return result

class HTMLTableSerializerProvider(ChunkingSerializerProvider):
    """Serializer provider that uses HTML for tables."""
    
    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=HTMLTableSerializer(),  # Use our custom HTML table serializer
        )


tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
max_tokens = tokenizer.model_max_length

custom_tokenizer: BaseTokenizer = HuggingFaceTokenizer(
    tokenizer=tokenizer,
    max_tokens=max_tokens,
)

converter = DocumentConverter()

chunker = HybridChunker(
    tokenizer=custom_tokenizer,
    serializer_provider=HTMLTableSerializerProvider(),
)


def save_doc_as_json(doc: DoclingDocument, json_path: str):
    """Save DoclingDocument as JSON file using model_dump()."""
    doc_dict = doc.model_dump()

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(doc_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Document saved as JSON to: {json_path}")


def parse_pdf(converter: DocumentConverter, pdf_path: str) -> DoclingDocument:
    start = time.time()
    doc = converter.convert(source=pdf_path).document
    elapsed = time.time() - start
    logger.info(f"Parsed document in {elapsed:.3f} seconds")
    return doc


def print_section_headers(doc: DoclingDocument) -> None:
    for text_item in doc.texts:
        if text_item.label == DocItemLabel.SECTION_HEADER:
            print(text_item.text)
            print("\n" + "-"*40)


def chunking(chunker: HybridChunker, doc: DoclingDocument) -> list[DocChunk]:
    start = time.time()
    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)
    elapsed = time.time() - start
    logger.info(f"Chunked document in {elapsed:.3f} seconds")

    return chunks


def print_extracted_chunks(chunks, tokenizer, chunker, save_path):
    with open(save_path, "w") as f:
        with redirect_stdout(f):
            for chunk_pos in range(len(chunks)):
                ctx_text = chunker.contextualize(chunk=chunks[chunk_pos])
                num_tokens = tokenizer.count_tokens(text=ctx_text)
                doc_items_refs = [it.self_ref for it in chunks[chunk_pos].meta.doc_items]
                print(
                    {
                        "first_element_page_number": chunks[chunk_pos].meta.doc_items[0].prov[0].page_no,
                        "last_element_page_number": chunks[chunk_pos].meta.doc_items[-1].prov[0].page_no,
                        "num_tokens": num_tokens,
                        "doc_items_refs": doc_items_refs,
                    }
                )
                print(ctx_text)
                print("\n" + "-"*40)
                print("\n\n")


def apply_chunking(pdf_path: str, save_doc_dir: str = None, print_titles: bool = False, save_chunks_dir: str = None) -> list[DocChunk]:

    logger.info(f"Semantic segmentation for given document (parsing)...")
    doc = parse_pdf(converter=converter, pdf_path=pdf_path)
    
    if save_doc_dir:
        os.makedirs(save_doc_dir, exist_ok=True)
        json_path = os.path.join(save_doc_dir, os.path.basename(pdf_path).split(".")[0] + "_doc.json")
        save_doc_as_json(doc=doc, json_path=json_path)

    if print_titles:
        logger.info("Extracted section headers:\n")
        print_section_headers(doc=doc)

    logger.info("Chunking...")
    chunks = chunking(chunker=chunker, doc=doc)

    if save_chunks_dir:
        os.makedirs(save_chunks_dir, exist_ok=True)
        save_path = os.path.join(save_chunks_dir, os.path.basename(pdf_path).split(".")[0] + "_chunks.txt")
        print_extracted_chunks(chunks=chunks, tokenizer=custom_tokenizer, chunker=chunker, save_path=save_path)

    return chunks


