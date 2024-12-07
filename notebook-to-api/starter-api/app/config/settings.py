import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set in environment variables")
    DOC_SOURCE_PATH: str = '../knowledgebase/'
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHUNK_SIZE: int = 1800
    CHUNK_OVERLAP: int = 300

settings = Settings()
