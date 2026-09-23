import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def load_documents(folder_path=config.DATA_DIR):
    """
    Goes through the given folder and loads all .pdf and .txt files.
    Returns a list of LangChain Document objects.
    """
    documents = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())

        else:
            print(f"Skipping {filename} (unsupported file type)")

    print(f"Loaded {len(documents)} document(s) from {folder_path}")
    return documents


def split_documents(documents):
    """
    Splits documents into smaller chunks so they can be embedded properly.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks
