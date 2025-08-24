import os
import sys

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any

import weaviate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompts import RAG_GENERATOR_SYSTEM_PROMPT


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

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
BASE_URL = os.getenv("BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ReferenceChunk(BaseModel):
    """Reference to a document chunk"""
    document_name: str = Field(description="Name of the document file")
    page_number: str = Field(description="Page number where the information is found")


class StructuredResponse(BaseModel):
    """Structured response from the RAG system"""
    answer: str = Field(description="The answer to the user question based on provided context")
    relevant_chunks: List[ReferenceChunk] = Field(description="List of relevant document chunks used for the answer")




class Generator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            base_url=BASE_URL,
            api_key=OPENAI_API_KEY
        )
        self.llm_with_structured_output = self.llm.with_structured_output(StructuredResponse)

    def augment_retrieved_context(self, retrieved_context: list[weaviate.collections.classes.internal.Object]) -> str:

        logger.debug("retrieved_context:\n{chunks}".format(
            chunks="\n".join([
                "filename: {filename}\nchunk title and content: {chunk_text_with_headings}".format(filename=item.properties["filename"], chunk_text_with_headings=item.properties["content"])
                for item in retrieved_context
            ])
        ))

        retrieved_context_prompt = ["## **CONTEXT** (Top-{top_k} Retrieved)".format(top_k=len(retrieved_context))]
        for i, chunk in enumerate(retrieved_context, start=1):
            src = chunk.properties.get("filename", "") 
            page = chunk.properties.get("first_element_page_number", "")
            text = (chunk.properties["content"] or "").strip()
            retrieved_context_prompt.append(f"<источник=\"{src}\" страница={page}>\n{text}\n")
        retrieved_context_prompt.append(
            "\n**ПРАВИЛА:**\n"
            "1) Дай строгий ответ опираясь только на фактические данные, предоставленные в **CONTEXT**.\n"
            "2) Для численных значений (суммы, количества, проценты) извлекай только чистое число без текста, пробелов и форматирования:\n"
            "   - Если сумма '120.309.860 тысяч тенге', ответ должен быть: 120309860000\n"
            "   - Если процент '15,5%', ответ должен быть: 15.5\n"
            "   - Если количество '25 человек', ответ должен быть: 25\n"
            "3) Для текстовых ответов (ФИО, названия, адреса) давай максимально краткий и точный ответ без лишних слов.\n"
            "4) Выдели только релевантные чанки документов, которые были использованы для ответа.\n"
            "5) Для каждого релевантного чанка извлеки 'document_name' (источник) и 'page_number' (номер страницы).\n"
            "6) Если предоставленный **CONTEXT** не содержит информации необходимой для ответа, честно скажи, что не знаешь ответ на данный вопрос.\n"
            "7) Отвечай в строгом JSON формате с полями 'answer' и 'relevant_chunks'."
        )
        return "\n".join(retrieved_context_prompt)


    def generate(self, user_query: str, retrieved_context: list[weaviate.collections.classes.internal.Object]) -> StructuredResponse:
        messages = [
            SystemMessage(content=RAG_GENERATOR_SYSTEM_PROMPT)
        ]

        if len(retrieved_context) > 0:
            messages.append(
                SystemMessage(content=self.augment_retrieved_context(retrieved_context))
            )

        messages.append(
            HumanMessage(content=user_query)
        )

        response = self.llm_with_structured_output.invoke(messages)
        return response


if __name__ == "__main__":
    user_query = "Привет, как дела?"
    retrieved_context = []

    generator = Generator()
    
    generated_answer = generator.generate(user_query, retrieved_context)
    logger.info(f"Answer: {generated_answer.answer}")
    logger.info(f"Relevant chunks: {generated_answer.relevant_chunks}")