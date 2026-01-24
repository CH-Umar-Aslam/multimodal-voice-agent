import os
from dotenv import load_dotenv

load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
if not (OPENROUTER_API_KEY or  OPENROUTER_BASE_URL  or GOOGLE_API_KEY):
    raise ValueError("Please set your OPENROUTER_API_KEY or OPENROUTER_BASE_URL  GOOGLE_API_KEY in .env")

def ingest_data():
    print("Starting Ingestion...")

    urls = [
        "https://www.sunmarke.com/",
        "https://www.sunmarke.com/admissions/process/",
        "https://www.sunmarke.com/admissions/fees/",
        "https://www.sunmarke.com/academics/curriculum/",
        "https://www.sunmarke.com/school-life/facilities/",
        "https://www.sunmarke.com/school-life/extra-curricular-activities/"
    ]

    print(f"Scraping {len(urls)} pages...")
    loader = WebBaseLoader(urls)
    docs = loader.load()

    print("Chunking content...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} document chunks.")

    print("Generating Embeddings & Vector Store...")



   
    # embeddings = OpenAIEmbeddings(
    # model="text-embedding-3-large",
    #     api_key=OPENROUTER_API_KEY,
    #     base_url="https://openrouter.ai/api/v1",
    # )

    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    max_retries=5
    )

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    print("Ingestion Complete! Vector store saved to 'faiss_index'.")

if __name__ == "__main__":
    ingest_data()
