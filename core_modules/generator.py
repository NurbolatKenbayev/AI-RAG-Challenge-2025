import os
import sys

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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




class Generator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            base_url=BASE_URL,
            api_key=OPENAI_API_KEY
        )


    def augment_retrieved_context(self, retrieved_context: list[weaviate.collections.classes.internal.Object]) -> str:

        logger.info("retrieved_context:\n{chunks}".format(
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
            retrieved_context_prompt.append(f"<D{i} источник=\"{src}\" страница={page}>\n{text}\n</D{i}>")
        retrieved_context_prompt.append(
            "**ПРАВИЛА:**\n"
            "1) Ответь опираясь только на фактические данные, предоставленные в **CONTEXT**.\n"
            "2) Ссылайся на источники информации по предоставленным [источник] и [страница] как [источник, стр. страница].\n"
            "3) Если предоставленный **CONTEXT** не содержит информации необходимой для ответа, честно скажи, что не знаешь ответ на данный вопрос, и попроси уточнить запрос, либо сформулировать другой."
        )
        return "\n".join(retrieved_context_prompt)


    def generate(self, user_query: str, retrieved_context: list[weaviate.collections.classes.internal.Object]) -> str:
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

        response = self.llm.invoke(messages)
        return response.content


if __name__ == "__main__":
    user_query = "Привет, как дела?"
    retrieved_context = []

    generator = Generator()
    
    generated_answer = generator.generate(user_query, retrieved_context)
    logger.info(generated_answer)