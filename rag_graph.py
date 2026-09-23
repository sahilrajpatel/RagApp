"""
Simple command line interface for the RAG app.

Usage:
    python main.py ingest              -> builds the vector store from data/
    python main.py ask "your question"  -> asks a question
"""
import sys

from src.load_data import load_documents, split_documents
from src.vector_store import create_vector_store
from src.rag_graph import ask_question


def ingest():
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)
    print("Done! You can now ask questions.")


def ask(question):
    answer = ask_question(question)
    print("\nAnswer:", answer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [ingest | ask \"your question\"]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        ingest()
    elif command == "ask":
        if len(sys.argv) < 3:
            print("Please provide a question. Example:")
            print('python main.py ask "What is this document about?"')
            sys.exit(1)
        ask(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [ingest | ask \"your question\"]")
