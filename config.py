from typing import Dict, Any, Optional
import os
from pydantic import BaseModel

class LLMConfig(BaseModel):
    """Configuration for LLM endpoints and settings."""
    endpoint: str
    api_key: Optional[str] = None
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    headers: Dict[str, str] = {}
    is_local: bool = False

class Config:
    """Main configuration class."""
    def __init__(self):
        self.researcher_llm = LLMConfig(
            endpoint=os.getenv("RESEARCHER_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
            api_key=os.getenv("RESEARCHER_LLM_API_KEY"),
            model=os.getenv("RESEARCHER_LLM_MODEL", "gpt-4"),
            temperature=float(os.getenv("RESEARCHER_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("RESEARCHER_LLM_MAX_TOKENS", "1000")),
            timeout=int(os.getenv("RESEARCHER_LLM_TIMEOUT", "30")),
            is_local=os.getenv("RESEARCHER_LLM_IS_LOCAL", "false").lower() == "true"
        )
        
        self.writer_llm = LLMConfig(
            endpoint=os.getenv("WRITER_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
            api_key=os.getenv("WRITER_LLM_API_KEY"),
            model=os.getenv("WRITER_LLM_MODEL", "gpt-4"),
            temperature=float(os.getenv("WRITER_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("WRITER_LLM_MAX_TOKENS", "2000")),
            timeout=int(os.getenv("WRITER_LLM_TIMEOUT", "30")),
            is_local=os.getenv("WRITER_LLM_IS_LOCAL", "false").lower() == "true"
        )

# Create a global config instance
config = Config() 