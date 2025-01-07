import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Settings:  # type: ignore
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set in environment variables")
    DOC_SOURCE_PATH: Optional[str] = "./knowledgebase/"
    EMBEDDING_MODEL: Optional[str] = "text-embedding-3-small"
    CHUNK_SIZE: int = 1800
    CHUNK_OVERLAP: int = 300


settings = Settings()
