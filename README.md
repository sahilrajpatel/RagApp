from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src import config

embeddings = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    api_key=config.OPENAI_API_KEY,
)


def create_vector_store(chunks):
    """
    Creates a Chroma vector store from document chunks and saves it to disk.
    Run this once whenever you add/change documents in the data folder.
    """
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.DB_DIR,
    )
    print(f"Vector store saved to {config.DB_DIR}")
    return db


def load_vector_store():
    """
    Loads the vector store that was already created.
    """
    return Chroma(
        embedding_function=embeddings,
        persist_directory=config.DB_DIR,
    )


def get_retriever():
    db = load_vector_store()
    return db.as_retriever(search_kwargs={"k": config.TOP_K})
