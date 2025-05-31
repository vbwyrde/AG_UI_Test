from typing import Dict, Any, Optional
import os
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        # Log all environment variables for debugging
        logger.debug("All environment variables:")
        for key, value in os.environ.items():
            logger.debug(f"{key}={value}")

        # Get and log the is_local values
        researcher_is_local = os.getenv("RESEARCHER_LLM_IS_LOCAL", "false")
        writer_is_local = os.getenv("WRITER_LLM_IS_LOCAL", "false")
        
        logger.debug(f"\nRaw is_local values:")
        logger.debug(f"RESEARCHER_LLM_IS_LOCAL={researcher_is_local}")
        logger.debug(f"WRITER_LLM_IS_LOCAL={writer_is_local}")
        
        # Convert to boolean, ensuring case-insensitive comparison
        is_local_researcher = str(researcher_is_local).lower() == "true"
        is_local_writer = str(writer_is_local).lower() == "true"
        
        logger.debug(f"\nParsed is_local values:")
        logger.debug(f"is_local_researcher={is_local_researcher}")
        logger.debug(f"is_local_writer={is_local_writer}")
        
        # Set default endpoints based on is_local flag
        researcher_endpoint = os.getenv("RESEARCHER_LLM_ENDPOINT")
        writer_endpoint = os.getenv("WRITER_LLM_ENDPOINT")
        
        logger.debug(f"\nEndpoint values:")
        logger.debug(f"RESEARCHER_LLM_ENDPOINT={researcher_endpoint}")
        logger.debug(f"WRITER_LLM_ENDPOINT={writer_endpoint}")
        
        if not researcher_endpoint:
            researcher_endpoint = "http://localhost:1234/v1/chat/completions" if is_local_researcher else "https://api.openai.com/v1/chat/completions"
        
        if not writer_endpoint:
            writer_endpoint = "http://localhost:1234/v1/chat/completions" if is_local_writer else "https://api.openai.com/v1/chat/completions"
        
        logger.debug(f"\nFinal endpoint values:")
        logger.debug(f"researcher_endpoint={researcher_endpoint}")
        logger.debug(f"writer_endpoint={writer_endpoint}")
        
        # Get model names
        researcher_model = os.getenv("RESEARCHER_LLM_MODEL", "Qwen_QwQ-32B-Q6_K_L" if is_local_researcher else "gpt-4")
        writer_model = os.getenv("WRITER_LLM_MODEL", "Qwen_QwQ-32B-Q6_K_L" if is_local_writer else "gpt-4")
        
        logger.debug(f"\nModel values:")
        logger.debug(f"researcher_model={researcher_model}")
        logger.debug(f"writer_model={writer_model}")
        
        self.researcher_llm = LLMConfig(
            endpoint=researcher_endpoint,
            api_key=None if is_local_researcher else os.getenv("RESEARCHER_LLM_API_KEY"),
            model=researcher_model,
            temperature=float(os.getenv("RESEARCHER_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("RESEARCHER_LLM_MAX_TOKENS", "1000")),
            timeout=int(os.getenv("RESEARCHER_LLM_TIMEOUT", "30")),
            is_local=is_local_researcher
        )
        
        self.writer_llm = LLMConfig(
            endpoint=writer_endpoint,
            api_key=None if is_local_writer else os.getenv("WRITER_LLM_API_KEY"),
            model=writer_model,
            temperature=float(os.getenv("WRITER_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("WRITER_LLM_MAX_TOKENS", "2000")),
            timeout=int(os.getenv("WRITER_LLM_TIMEOUT", "30")),
            is_local=is_local_writer
        )
        
        logger.debug(f"\nFinal LLM Configurations:")
        logger.debug(f"Researcher LLM Config: {self.researcher_llm}")
        logger.debug(f"Writer LLM Config: {self.writer_llm}")

# Create a global config instance
config = Config() 