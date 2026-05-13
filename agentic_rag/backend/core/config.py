from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Critic-Guided Agentic RAG"
    CHROMA_PERSIST_DIR: str = "../db" # Relative to where the script is run
    MODEL_PATH: str = "../Phi-3-mini-4k-instruct-q4.gguf" 
    MAX_RETRIES: int = 3
    
    class Config:
        env_file = ".env"

settings = Settings()
