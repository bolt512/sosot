import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    # Mattermost
    mm_url: str
    mm_port: int
    mm_token: str
    mm_scheme: str
    bot_name: str

    # Ollama
    ollama_base_url: str
    llm_model: str
    embedding_model: str

    # ChromaDB
    db_path: str
    data_path: str

    # RAG
    chunk_size: int
    chunk_overlap: int
    retriever_k: int

    # History
    max_history: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            mm_url=os.getenv("MM_URL", "localhost"),
            mm_port=int(os.getenv("MM_PORT", "8065")),
            mm_token=os.getenv("MM_TOKEN", ""),
            mm_scheme=os.getenv("MM_SCHEME", "http"),
            bot_name=os.getenv("BOT_NAME", "sosot"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            llm_model=os.getenv("LLM_MODEL", "llama3"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            db_path=os.getenv("DB_PATH", "./vector_db"),
            data_path=os.getenv("DATA_PATH", "./data"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            retriever_k=int(os.getenv("RETRIEVER_K", "3")),
            max_history=int(os.getenv("MAX_HISTORY", "10")),
        )
