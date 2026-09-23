import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

DATA_DIR = "data"
DB_DIR = "vector_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
