import os
from dotenv import load_dotenv

load_dotenv()
host = os.getenv("PG_VECTOR_HOST")
user = os.getenv("PG_VECTOR_USER")
password = os.getenv("PG_VECTOR_PASSWORD")
COLLECTION_NAME = os.getenv("PG_DATABASE", "postgres")
CONNECTION_STRING = f"postgresql+psycopg://{os.getenv('PG_USER', 'postgres')}:{os.getenv('PG_PASSWORD', '')}@{os.getenv('PG_HOST', 'localhost')}:{os.getenv('PG_PORT', '5432')}/{os.getenv('PG_DATABASE', 'postgres')}"
