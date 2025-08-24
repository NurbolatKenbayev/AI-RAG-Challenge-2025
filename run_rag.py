import os
import sys

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
import pandas as pd
import json
import glob
from functools import reduce
import operator
import time

from weaviate.classes.query import Filter

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
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
BASE_URL = os.getenv("BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



class RAGPipeline:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            base_url=BASE_URL,
            api_key=OPENAI_API_KEY
        )


    def extract_company_name(self, user_query: str) -> list:
        """
        Extract all relevant company names from the user query using LLM.
        
        Args:
            user_query (str): The user's question
            
        Returns:
            list: List of company names that match the query. Empty list if no matches found.
        """
        # Get list of available companies from the dataset directory
        dataset_path = "/Users/nurbolatkenbayev/Desktop/work_dir/68a58892b4058539485342/Dataset"
        try:
            company_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        except Exception as e:
            logger.error(f"Error reading dataset directory: {e}")
            return []
        
        # Create prompt for LLM
        companies_list = "\n".join([f"- {company}" for company in company_dirs])
        
        prompt = f"""You are tasked with identifying ALL companies mentioned in a user question.

        Available companies:
        {companies_list}

        User question: "{user_query}"

        Instructions:
        1. Analyze the user question to identify ALL company names, abbreviations, or references
        2. Match ALL identified companies to ones from the available companies list above
        3. Return ALL matching company names, each on a separate line
        4. Use ONLY the exact company names from the list above
        5. If no companies are found, return "None"
        6. Do not provide any explanation or additional text
        7. Example format if multiple companies found:
           AsiaAgroFood
           Maten Petroleum

        Company names:"""

        try:
            response = self.llm.invoke(prompt)
            extracted_text = response.content.strip()
            
            # Parse the response to extract company names
            if extracted_text.lower() == "none":
                logger.info("No companies found in query")
                return []
            
            # Split by lines and clean up
            potential_companies = [line.strip() for line in extracted_text.split('\n') if line.strip()]
            
            # Validate that all extracted companies are in our list
            valid_companies = []
            for company in potential_companies:
                if company in company_dirs:
                    valid_companies.append(company)
                else:
                    logger.warning(f"LLM returned company '{company}' not in available list. Skipping.")
            
            logger.info(f"Extracted companies: {valid_companies}")
            return valid_companies
                
        except Exception as e:
            logger.error(f"Error in LLM call for company extraction: {e}")
            return []


    def rag_query(self, user_query: str, retriever: Retriever, generator: Generator, top_k: int = 5, alpha: float = 0.5):
        try:
            collection = retriever.get_collection(WEAVIATE_COLLECTION_NAME)
            if collection is None:
                logger.error(f"Collection {WEAVIATE_COLLECTION_NAME} not found")
                return "", []

            # Extract company names from the user query to squeeze search space
            company_names = self.extract_company_name(user_query)
            logger.info(f"Company names: {company_names}")
            
            dynamic_or_filter = None
            if company_names:
                data_dir = os.path.expanduser("~/Desktop/work_dir/68a58892b4058539485342/Dataset")
                all_file_names = []
                
                # Collect all PDF files from all identified companies
                for company_name in company_names:
                    dir_company = [item for item in glob.glob(data_dir+"/*") if os.path.basename(item) == company_name]
                    if len(dir_company) == 1:
                        dir_company = dir_company[0]
                        file_names = [os.path.basename(item) for item in glob.glob(dir_company+"/*.pdf")]
                        all_file_names.extend(file_names)
                
                # Create filter for all files from all identified companies
                if all_file_names:
                    filters_list = [Filter.by_property("filename").equal(file_name) for file_name in all_file_names]
                    dynamic_or_filter = reduce(operator.or_, filters_list)
                    logger.info(f"Created filter for {len(all_file_names)} files from {len(company_names)} companies")

            # Search for relevant chunks in the Weaviate collection
            weaviate_objects = retriever.hybrid_search(
                query_text=user_query, 
                collection=collection,
                top_k=top_k,
                alpha=alpha,
                filter_condition=dynamic_or_filter
            )
            relevant_chunks = [
                {
                    "document_name": obj.properties["filename"], 
                    "page_number": obj.properties["first_element_page_number"]
                }  
                for obj in weaviate_objects
            ]

            try:
                structured_response = generator.generate(
                    user_query=user_query, 
                    retrieved_context=weaviate_objects
                )
                return structured_response.answer, structured_response.relevant_chunks
            except Exception as e:
                logger.error(f"Error in generation step: {e}")
                # Return empty answer but keep the retrieved chunks info for debugging
                return "", []
                
        except Exception as e:
            logger.error(f"Error in rag_query: {e}")
            return "", []


    def process_query_set(self, file_path: str, retriever: Retriever, generator: Generator, top_k: int = 3, alpha: float = 0.5):
        df = pd.read_excel(file_path)
        answers = []
        
        for index, row in df.iterrows():
            try:
                logger.info(f"Processing question id={row['id']} ({index+1}/{len(df)})")
                logger.info(f"Question: {row['full_question']}")
                
                answer, relevant_chunks = self.rag_query(row["full_question"], retriever, generator, top_k=top_k, alpha=alpha)
                
                # Handle answer type conversion safely
                type_str = row["answer_type"]
                generated_answer = ""
                
                if answer:  # Only try type conversion if we have an answer
                    try:
                        if type_str == "float":
                            answer = answer.replace(",", ".")
                        generated_answer = eval(f"{type_str}('{answer}')")
                    except Exception as e:
                        logger.warning(f"Could not evaluate answer to type {type_str}: {e}. Using raw answer.")
                        generated_answer = answer
                else:
                    logger.warning(f"No answer generated for question id={row['id']}")
                    generated_answer = ""

                logger.info(f"Generated answer: {generated_answer}")
                logger.info(f"Relevant chunks count: {len(relevant_chunks)}")
                
                # Safely process relevant_chunks
                processed_chunks = []
                if relevant_chunks:
                    try:
                        processed_chunks = [{
                            "document_name": item.document_name,
                            "page_number": int(item.page_number)
                        } for item in relevant_chunks]
                    except Exception as e:
                        logger.error(f"Error processing relevant chunks: {e}")
                        processed_chunks = []
                
                answers.append({
                    "question_id": row["id"],
                    "relevant_chunks": processed_chunks,
                    "answer": generated_answer
                })
                
            except Exception as e:
                logger.error(f"Error processing question id={row['id']}: {e}")
                # Add fallback answer even if processing completely fails
                answers.append({
                    "question_id": row["id"],
                    "relevant_chunks": [],
                    "answer": ""
                })
            
            # break

        return answers



if __name__ == "__main__":
    retriever = Retriever()
    generator = Generator()
    rag_pipeline = RAGPipeline()
    
    top_k = 10
    alpha = 1.0


    # # Test company extraction
    # test_query = "Кто главный бухгалтер в АО \"AsiaAgroFood\" (ФИО)?" # id=14
    # extracted_companies = rag_pipeline.extract_company_name(test_query)
    # print(f"Test query: {test_query}")
    # print(f"Extracted companies: {extracted_companies}")
    # 
    # # Test with multiple companies
    # test_query_multi = "Compare AsiaAgroFood and Maten Petroleum financial performance"
    # extracted_companies_multi = rag_pipeline.extract_company_name(test_query_multi)
    # print(f"Multi-company test query: {test_query_multi}")
    # print(f"Extracted companies: {extracted_companies_multi}")
    
    # # user_query = "Кто главный бухгалтер в АО \"AsiaAgroFood\" (ФИО)?" # id=14
    # user_query = "Какая сумма в тенге налоговых убытков была у группы компаний АО «Матен Петролеум» на момент 31 декабря 2024 года?" # id=9
    # answer, relevant_chunks = rag_pipeline.rag_query(user_query, retriever, generator, top_k=top_k, alpha=alpha)
    # print()
    # print(f"Generated answer: {answer}")
    # print(f"Relevant chunks: {relevant_chunks}")

    time_start = time.time()
    # file_path = '../68a58892b4058539485342/questions_public.xlsx'
    file_path = '../68a86b9f5d944822407386.xlsx'
    answers = rag_pipeline.process_query_set(file_path, retriever, generator, top_k=top_k, alpha=alpha)
    time_end = time.time()  
    print(f"Time taken: {time_end - time_start} seconds")
    with open("answers.json", "w", encoding='utf-8') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)


    # Close Weaviate client
    retriever.close_client()
