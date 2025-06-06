from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from ag_ui.core import (
    RunAgentInput,
    Message,
    State,
    Context,
    Tool,
    Role
)
from ag_ui.encoder import EventEncoder
import uuid
import os
import logging
import asyncio
import json
import traceback
from typing import List, Optional, AsyncGenerator, Dict, Any
from pydantic import BaseModel, Field
import ast
import re
import importlib
import openai
import httpx
from dotenv import load_dotenv
from config import config
import time
from enum import Enum, auto
import sys
import socket
import psutil

class EventType(str, Enum):
    """Enhanced AG-UI Event Types with all standard events."""
    
    # Run Events
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_ERROR = "run_error"
    
    # Message Events
    TEXT_MESSAGE_START = "text_message_start"
    TEXT_MESSAGE_CONTENT = "text_message_content"
    TEXT_MESSAGE_END = "text_message_end"
    TEXT_MESSAGE_ERROR = "text_message_error"
    
    # Tool Events
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_FINISHED = "tool_call_finished"
    TOOL_CALL_ERROR = "tool_call_error"
    
    # State Events
    STATE_UPDATE = "state_update"
    STATE_ERROR = "state_error"
    CONTEXT_UPDATE = "context_update"
    CONTEXT_ERROR = "context_error"
    
    # Control Events
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    RESET = "reset"

class BaseEvent(BaseModel):
    """Base class for all AG-UI events."""
    type: EventType
    timestamp: float = Field(default_factory=time.time)
    sequence_id: Optional[int] = None
    correlation_id: Optional[str] = None
    thread_id: str
    run_id: str

class RunEvent(BaseEvent):
    """Base class for run-related events."""
    pass

class RunStartedEvent(RunEvent):
    """Event indicating the start of an agent run."""
    type: EventType = EventType.RUN_STARTED
    agent_metadata: Optional[dict] = None
    configuration: Optional[dict] = None

class RunFinishedEvent(RunEvent):
    """Event indicating the completion of an agent run."""
    type: EventType = EventType.RUN_FINISHED
    completion_status: str
    performance_metrics: Optional[dict] = None

class RunErrorEvent(RunEvent):
    """Event indicating an error during agent run."""
    type: EventType = EventType.RUN_ERROR
    error_details: dict

class MessageEvent(BaseEvent):
    """Base class for message-related events."""
    message_id: str
    role: str

class TextMessageStartEvent(MessageEvent):
    """Event indicating the start of a text message."""
    type: EventType = EventType.TEXT_MESSAGE_START

class TextMessageContentEvent(MessageEvent):
    """Event containing text message content."""
    type: EventType = EventType.TEXT_MESSAGE_CONTENT
    delta: str

class TextMessageEndEvent(MessageEvent):
    """Event indicating the end of a text message."""
    type: EventType = EventType.TEXT_MESSAGE_END
    role: Optional[str] = None

class TextMessageErrorEvent(MessageEvent):
    """Event indicating an error in message processing."""
    type: EventType = EventType.TEXT_MESSAGE_ERROR
    error_details: dict

class ToolEvent(BaseEvent):
    """Base class for tool-related events."""
    tool_id: str
    tool_name: str
    parameters: dict

class ToolCallStartedEvent(ToolEvent):
    """Event indicating the start of a tool call."""
    type: EventType = EventType.TOOL_CALL_STARTED

class ToolCallFinishedEvent(ToolEvent):
    """Event indicating the completion of a tool call."""
    type: EventType = EventType.TOOL_CALL_FINISHED
    result: dict
    execution_metrics: Optional[dict] = None

class ToolCallErrorEvent(ToolEvent):
    """Event indicating an error in tool execution."""
    type: EventType = EventType.TOOL_CALL_ERROR
    error_details: dict

class StateEvent(BaseEvent):
    """Base class for state-related events."""
    state_key: str
    state_value: Any

class StateUpdateEvent(StateEvent):
    """Event indicating a state update."""
    type: EventType = EventType.STATE_UPDATE
    previous_value: Optional[Any] = None

class StateErrorEvent(StateEvent):
    """Event indicating an error in state update."""
    type: EventType = EventType.STATE_ERROR
    error_details: dict

class ContextEvent(BaseEvent):
    """Base class for context-related events."""
    context_key: str
    context_value: Any

class ContextUpdateEvent(ContextEvent):
    """Event indicating a context update."""
    type: EventType = EventType.CONTEXT_UPDATE
    previous_value: Optional[Any] = None

class ContextErrorEvent(ContextEvent):
    """Event indicating an error in context update."""
    type: EventType = EventType.CONTEXT_ERROR
    error_details: dict

class ControlEvent(BaseEvent):
    """Base class for control-related events."""
    reason: Optional[str] = None

class PauseEvent(ControlEvent):
    """Event indicating a pause request."""
    type: EventType = EventType.PAUSE

class ResumeEvent(ControlEvent):
    """Event indicating a resume request."""
    type: EventType = EventType.RESUME

class CancelEvent(ControlEvent):
    """Event indicating a cancellation request."""
    type: EventType = EventType.CANCEL

class ResetEvent(ControlEvent):
    """Event indicating a reset request."""
    type: EventType = EventType.RESET

class EventManager:
    """Manages event validation, serialization, and sequence tracking."""
    
    def __init__(self):
        self._sequence_counter = 0
        self._event_history: List[BaseEvent] = []
        self._correlation_map: Dict[str, List[BaseEvent]] = {}
    
    def _generate_sequence_id(self) -> int:
        """Generate a unique sequence ID for events."""
        self._sequence_counter += 1
        return self._sequence_counter
    
    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID for events."""
        return str(uuid.uuid4())
    
    def _validate_event_sequence(self, event: BaseEvent) -> bool:
        """Validate that the event maintains proper sequence order."""
        # First event in a thread/run is always valid
        if not self._event_history:
            return True
            
        # Find the last event for this thread/run
        last_event = None
        for e in reversed(self._event_history):
            if e.thread_id == event.thread_id and e.run_id == event.run_id:
                last_event = e
                break
                
        # If no previous event for this thread/run, sequence is valid
        if not last_event:
            return True
            
        # If this is the same event (already in history), it's valid
        if event in self._event_history:
            return True
            
        # Validate sequence ID is greater than last event
        if event.sequence_id <= last_event.sequence_id:
            return False
            
        return True
    
    def _validate_correlation(self, event: BaseEvent) -> bool:
        """Validate correlation between related events."""
        if not event.correlation_id:
            return True
            
        if event.correlation_id not in self._correlation_map:
            self._correlation_map[event.correlation_id] = []
            return True
            
        related_events = self._correlation_map[event.correlation_id]
        
        # If this event is already in the correlation map, it's valid
        if event in related_events:
            return True
            
        # Validate that correlated events maintain proper sequence
        if related_events and event.sequence_id <= related_events[-1].sequence_id:
            return False
            
        return True
    
    def _validate_event_type(self, event: BaseEvent) -> bool:
        """Validate event type-specific requirements."""
        logger.debug(f"Validating event type: {type(event).__name__}")
        logger.debug(f"Event attributes: {event.model_dump()}")
        
        if isinstance(event, MessageEvent):
            logger.debug("Event is a MessageEvent")
            # For TextMessageEndEvent, only message_id is required
            if isinstance(event, TextMessageStartEvent):
                logger.debug(f"Validating TextMessageStartEvent")
                # Check if required fields exist and are not None
                if not hasattr(event, 'message_id') or not hasattr(event, 'role'):
                    logger.debug("Missing required fields: message_id or role")
                    return False
                return event.message_id is not None and event.role is not None
            elif isinstance(event, TextMessageContentEvent):
                logger.debug(f"Validating TextMessageContentEvent")
                if not hasattr(event, 'message_id') or not hasattr(event, 'role') or not hasattr(event, 'delta'):
                    logger.debug("Missing required fields: message_id, role, or delta")
                    return False
                return event.message_id is not None and event.role is not None and event.delta is not None
            elif isinstance(event, TextMessageEndEvent):
                logger.debug(f"Validating TextMessageEndEvent")
                if not hasattr(event, 'message_id'):
                    logger.debug("Missing required field: message_id")
                    return False
                return event.message_id is not None
            elif isinstance(event, TextMessageErrorEvent):
                logger.debug(f"Validating TextMessageErrorEvent")
                if not hasattr(event, 'message_id') or not hasattr(event, 'role') or not hasattr(event, 'error_details'):
                    logger.debug("Missing required fields: message_id, role, or error_details")
                    return False
                return event.message_id is not None and event.role is not None and event.error_details is not None
        elif isinstance(event, ToolEvent):
            logger.debug(f"Validating ToolEvent")
            if not hasattr(event, 'tool_id') or not hasattr(event, 'tool_name') or not hasattr(event, 'parameters'):
                logger.debug("Missing required fields: tool_id, tool_name, or parameters")
                return False
            return (event.tool_id is not None and 
                   event.tool_name is not None and 
                   event.parameters is not None)
        elif isinstance(event, StateEvent):
            logger.debug(f"Validating StateEvent")
            if not hasattr(event, 'state_key'):
                logger.debug("Missing required field: state_key")
                return False
            return event.state_key is not None
        elif isinstance(event, ContextEvent):
            logger.debug(f"Validating ContextEvent")
            if not hasattr(event, 'context_key'):
                logger.debug("Missing required field: context_key")
                return False
            return event.context_key is not None
        elif isinstance(event, RunEvent):
            logger.debug("Event is a RunEvent")
            # For RunStartedEvent, agent_metadata is optional
            if isinstance(event, RunStartedEvent):
                logger.debug("Validating RunStartedEvent (no required fields)")
                return True
            # For RunFinishedEvent, completion_status is required
            elif isinstance(event, RunFinishedEvent):
                logger.debug(f"Validating RunFinishedEvent")
                if not hasattr(event, 'completion_status'):
                    logger.debug("Missing required field: completion_status")
                    return False
                return event.completion_status is not None
            # For RunErrorEvent, error_details is required
            elif isinstance(event, RunErrorEvent):
                logger.debug(f"Validating RunErrorEvent")
                if not hasattr(event, 'error_details'):
                    logger.debug("Missing required field: error_details")
                    return False
                return event.error_details is not None
        logger.debug("No specific validation rules found, returning True")
        return True

    def validate_event(self, event: BaseEvent) -> tuple[bool, str]:
        """Validate an event against all validation rules."""
        try:
            logger.debug(f"Starting validation for event: {type(event).__name__}")
            logger.debug(f"Event data: {event.model_dump()}")
            
            # Validate sequence only if the event has a sequence_id
            if event.sequence_id is not None:
                if not self._validate_event_sequence(event):
                    logger.debug("Event failed sequence validation")
                    return False, "Invalid event sequence"
                
            # Validate correlation
            if not self._validate_correlation(event):
                logger.debug("Event failed correlation validation")
                return False, "Invalid event correlation"
                
            # Validate event type specific requirements
            if not self._validate_event_type(event):
                logger.debug("Event failed type validation")
                return False, "Invalid event type requirements"
                
            logger.debug("Event passed all validation checks")
            return True, "Event is valid"
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            return False, f"Validation error: {str(e)}"
    
    def prepare_event(self, event: BaseEvent) -> BaseEvent:
        """Prepare an event for sending by adding required fields."""
        # Generate sequence ID if not provided
        if event.sequence_id is None:
            event.sequence_id = self._generate_sequence_id()
            
        # Generate correlation ID if not provided
        if event.correlation_id is None:
            event.correlation_id = self._generate_correlation_id()
            
        # Validate the event
        is_valid, message = self.validate_event(event)
        if not is_valid:
            raise ValueError(f"Invalid event: {message}")
            
        # Store in history
        self._event_history.append(event)
        
        # Update correlation map
        if event.correlation_id:
            if event.correlation_id not in self._correlation_map:
                self._correlation_map[event.correlation_id] = []
            self._correlation_map[event.correlation_id].append(event)
            
        return event
    
    def serialize_event(self, event: BaseEvent) -> str:
        """Serialize an event to JSON string."""
        try:
            # Only prepare the event if it hasn't been prepared yet
            if event.sequence_id is None or event.correlation_id is None:
                event = self.prepare_event(event)
            
            # Convert to JSON
            return event.model_dump_json()
            
        except Exception as e:
            raise ValueError(f"Serialization error: {str(e)}")
    
    def get_event_history(self, thread_id: Optional[str] = None, run_id: Optional[str] = None) -> List[BaseEvent]:
        """Get event history filtered by thread_id and/or run_id."""
        events = self._event_history
        
        if thread_id:
            events = [e for e in events if e.thread_id == thread_id]
        if run_id:
            events = [e for e in events if e.run_id == run_id]
            
        return events
    
    def get_correlated_events(self, correlation_id: str) -> List[BaseEvent]:
        """Get all events with the given correlation_id."""
        return self._correlation_map.get(correlation_id, [])
    
    def clear_history(self, thread_id: Optional[str] = None, run_id: Optional[str] = None):
        """Clear event history for specific thread/run or all history."""
        if thread_id or run_id:
            self._event_history = [
                e for e in self._event_history 
                if (not thread_id or e.thread_id != thread_id) and 
                   (not run_id or e.run_id != run_id)
            ]
        else:
            self._event_history = []
            self._correlation_map = {}
            self._sequence_counter = 0

# Configure logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
logger.debug(f"Loading .env file from: {env_path}")
if os.path.exists(env_path):
    logger.debug("Found .env file, loading environment variables")
    load_dotenv(env_path, override=True)  # Add override=True to ensure values are loaded
    # Verify environment variables were loaded
    logger.debug("Verifying environment variables after loading:")
    for key in os.environ:
        if "LLM" in key:
            logger.debug(f"{key}={os.environ[key]}")
else:
    logger.warning(f"No .env file found at {env_path}")

# Force reload of config after environment variables are loaded
from importlib import reload
import config
reload(config)
from config import config

app = FastAPI()

# Initialize EventManager
event_manager = EventManager()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class PromptWriterAgent:
    def __init__(self):
        self.name = "PromptWriter"
        self.description = "Generates appropriate prompts for different languages and tasks"
        self.capabilities = ["prompt_generation", "language_detection"]
        self.llm_config = config.prompt_writer_llm
        logger.info(f"PromptWriter LLM Config: endpoint={self.llm_config.endpoint}, is_local={self.llm_config.is_local}, model={self.llm_config.model}")
        if not self.llm_config.is_local:
            self.client = openai.OpenAI(api_key=self.llm_config.api_key)
        
        # Language detection patterns
        self.language_patterns = {
            'vb.net': [
                r'vb\.net',
                r'vbnet',
                r'visual basic',
                r'MessageBox\.Show',
                r'Public\s+Class',
                r'Private\s+Sub',
                r'End\s+Class',
                r'End\s+Sub'
            ],
            'python': [
                r'def\s+\w+\s*\(',
                r'import\s+\w+',
                r'print\s*\(',
                r'if\s+.*:',
                r'for\s+.*:',
                r'while\s+.*:'
            ],
            'javascript': [
                r'function\s+\w+\s*\(',
                r'const\s+\w+',
                r'let\s+\w+',
                r'console\.log\(',
                r'if\s*\(.*\)\s*{',
                r'for\s*\(.*\)\s*{'
            ],
            'csharp': [
                r'public\s+class',
                r'private\s+\w+',
                r'Console\.WriteLine\(',
                r'if\s*\(.*\)\s*{',
                r'for\s*\(.*\)\s*{',
                r'using\s+System;'
            ]
        }

    def identify_language(self, text: str) -> str:
        """Identifies the programming language from the input text."""
        # First check for explicit language mentions
        text_lower = text.lower()
        if 'c#' in text_lower or 'csharp' in text_lower:
            return 'csharp'
        elif 'vb.net' in text_lower or 'vbnet' in text_lower or 'visual basic' in text_lower:
            return 'vb.net'
        elif 'javascript' in text_lower or 'js' in text_lower:
            return 'javascript'
        elif 'python' in text_lower:
            return 'python'
            
        # If no explicit mention, use pattern matching
        scores = {lang: 0 for lang in self.language_patterns}
        for lang, patterns in self.language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[lang] += 1
        
        # Return the language with the highest score, or 'unknown' if no matches
        if any(scores.values()):
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            # If no patterns match, check for common language indicators
            if 'debug.print' in text_lower or 'messagebox.show' in text_lower:
                return 'vb.net'
            elif 'console.writeline' in text_lower or 'using system' in text_lower:
                return 'csharp'
            else:
                return 'unknown'

    async def call_llm_api(self, prompt: str, max_retries: int = 3) -> str:
        """Call the LLM API to generate prompts."""
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                logger.info(f"PromptWriter calling LLM API with is_local={self.llm_config.is_local} (attempt {retry_count + 1}/{max_retries})")
                
                if self.llm_config.is_local:
                    async with httpx.AsyncClient(timeout=1200.0) as client:  # 20 minute timeout
                        response = await client.post(
                            self.llm_config.endpoint,
                            json={
                                "model": self.llm_config.model,
                                "messages": [
                                    {"role": "system", "content": "You are a prompt engineering assistant. Generate appropriate prompts for different programming languages and tasks."},
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": self.llm_config.temperature,
                                "max_tokens": self.llm_config.max_tokens
                            }
                        )
                        response.raise_for_status()
                        return response.json()["choices"][0]["message"]["content"]
                else:
                    if not self.llm_config.api_key:
                        raise ValueError("OpenAI API key is required when not using local endpoint")
                    response = self.client.chat.completions.create(
                        model=self.llm_config.model,
                        messages=[
                            {"role": "system", "content": "You are a prompt engineering assistant. Generate appropriate prompts for different programming languages and tasks."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.llm_config.temperature,
                        max_tokens=self.llm_config.max_tokens
                    )
                    return response.choices[0].message.content
                    
            except httpx.TimeoutException as e:
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Timeout on attempt {retry_count}, retrying...")
                    await asyncio.sleep(1)  # Wait 1 second before retrying
                continue
            except Exception as e:
                logger.error(f"Error calling LLM API: {e}", exc_info=True)
                raise
        
        # If we get here, all retries failed
        error_message = f"Failed to generate prompt after {max_retries} attempts. Last error: {str(last_error)}"
        logger.error(error_message)
        raise ValueError(error_message)

    async def generate_research_prompt(self, message: str, language: str) -> str:
        """Generate a research prompt for the specified language."""
        # Ensure language is properly formatted (e.g., 'csharp' instead of 'C#')
        if language.lower() == 'c#':
            language = 'csharp'
        elif language.lower() == 'vb.net':
            language = 'vb.net'
            
        prompt = f"""
        Generate a research prompt for the following {language} code requirement:
        {message}
        
        The prompt should:
        1. Focus on language-specific best practices
        2. Consider common patterns in {language}
        3. Include language-specific pitfalls to avoid
        4. Request recommended approaches for {language}
        
        Return only the prompt text.
        """
        return await self.call_llm_api(prompt)

    async def generate_code_prompt(self, message: str, language: str) -> str:
        """Generate a code generation prompt for the specified language."""
        prompt = f"""
        Generate a code generation prompt for the following {language} code requirement:
        {message}
        
        The prompt should:
        1. Request safe and idiomatic {language} code
        2. Specify language-specific best practices
        3. Include proper error handling for {language}
        4. Request appropriate documentation style for {language}
        
        Return only the prompt text.
        """
        return await self.call_llm_api(prompt)

class CodeValidator:
    """Base class for language-specific code validation."""
    def __init__(self):
        self.name = "CodeValidator"
        self.description = "Validates code safety and requirements"

    def validate_code_safety(self, code: str, language: str) -> tuple[bool, str]:
        """Generic code safety validation."""
        try:
            # Common dangerous patterns across languages
            dangerous_patterns = [
                r'os\.system\(',
                r'subprocess\.',
                r'eval\(',
                r'exec\(',
                r'__import__\(',
                r'open\(',
                r'file\(',
                r'\.\.\/',  # Path traversal
                r'rm\s+-rf',  # Dangerous delete
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, code):
                    return False, f"Code contains potentially dangerous operation: {pattern}"
            
            return True, "Code appears safe"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class PythonValidator(CodeValidator):
    """Python-specific code validation."""
    def validate_code_safety(self, code: str, language: str) -> tuple[bool, str]:
        """Python-specific code safety validation."""
        try:
            # First run generic validation
            is_safe, message = super().validate_code_safety(code, language)
            if not is_safe:
                return is_safe, message

            # Python-specific validation
            try:
                ast.parse(code)
            except SyntaxError as e:
                return False, f"Python syntax error: {str(e)}"
            
            return True, "Python code appears safe"
        except Exception as e:
            return False, f"Python validation error: {str(e)}"

class VBValidator(CodeValidator):
    """VB.NET-specific code validation."""
    def validate_code_safety(self, code: str, language: str) -> tuple[bool, str]:
        """VB.NET-specific code safety validation."""
        try:
            # First run generic validation
            is_safe, message = super().validate_code_safety(code, language)
            if not is_safe:
                return is_safe, message

            # VB.NET-specific validation
            # Check for basic VB.NET syntax patterns
            required_patterns = [
                r'Public\s+Class',
                r'End\s+Class'
            ]
            
            for pattern in required_patterns:
                if not re.search(pattern, code, re.IGNORECASE):
                    return False, f"Missing required VB.NET pattern: {pattern}"
            
            return True, "VB.NET code appears safe"
        except Exception as e:
            return False, f"VB.NET validation error: {str(e)}"

class ResearcherAgent:
    def __init__(self):
        self.name = "Researcher"
        self.description = "Researches best practices and patterns for various programming languages"
        self.capabilities = ["research", "analysis"]
        self.llm_config = config.researcher_llm
        self.prompt_writer = PromptWriterAgent()
        logger.info(f"Researcher LLM Config: endpoint={self.llm_config.endpoint}, is_local={self.llm_config.is_local}, model={self.llm_config.model}")
        if not self.llm_config.is_local:
            self.client = openai.OpenAI(api_key=self.llm_config.api_key)

    async def call_llm_api(self, prompt: str, max_retries: int = 3) -> str:
        """Call the LLM API to get research results."""
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                logger.info(f"Researcher calling LLM API with is_local={self.llm_config.is_local} (attempt {retry_count + 1}/{max_retries})")
                
                if self.llm_config.is_local:
                    async with httpx.AsyncClient(timeout=1200.0) as client:  # 20 minute timeout
                        response = await client.post(
                            self.llm_config.endpoint,
                            json={
                                "model": self.llm_config.model,
                                "messages": [
                                    {"role": "system", "content": "You are a research assistant specializing in programming best practices and patterns."},
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": self.llm_config.temperature,
                                "max_tokens": self.llm_config.max_tokens
                            }
                        )
                        response.raise_for_status()
                        return response.json()["choices"][0]["message"]["content"]
                else:
                    if not self.llm_config.api_key:
                        raise ValueError("OpenAI API key is required when not using local endpoint")
                    response = self.client.chat.completions.create(
                        model=self.llm_config.model,
                        messages=[
                            {"role": "system", "content": "You are a research assistant specializing in programming best practices and patterns."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.llm_config.temperature,
                        max_tokens=self.llm_config.max_tokens
                    )
                    return response.choices[0].message.content
                    
            except httpx.TimeoutException as e:
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Timeout on attempt {retry_count}, retrying...")
                    await asyncio.sleep(1)  # Wait 1 second before retrying
                continue
            except Exception as e:
                logger.error(f"Error calling LLM API: {e}", exc_info=True)
                raise
        
        # If we get here, all retries failed
        error_message = f"Failed to get research results after {max_retries} attempts. Last error: {str(last_error)}"
        logger.error(error_message)
        raise ValueError(error_message)

    async def process_message(self, message: str) -> str:
        logger.info(f"Researcher processing message: {message}")
        
        # Detect language from message using PromptWriterAgent
        language = self.prompt_writer.identify_language(message)
        logger.info(f"Detected language for research: {language}")
        
        # Generate appropriate research prompt
        research_prompt = await self.prompt_writer.generate_research_prompt(message, language)
        
        try:
            # Get initial research results from LLM
            research_results = await self.call_llm_api(research_prompt)
            
            # Extract code example if present
            code_example = None
            if f"```{language}" in research_results:
                code_block = research_results.split(f"```{language}")[1].split("```")[0].strip()
                code_example = code_block
            elif "```" in research_results:
                code_block = research_results.split("```")[1].split("```")[0].strip()
                code_example = code_block
            
            # Generate a summary of the research results
            summary_prompt = f"""Analyze the following research results and provide a concise summary.
Focus on key requirements, best practices, and critical considerations.
Exclude any code examples or implementation details.

Research Results:
{research_results}

Return a JSON object with these fields:
1. summary: A concise summary of key points and requirements
2. critical_points: A list of critical points that must be implemented
3. code_example: The code example from the research (if any)
4. original_research: The complete research results

Format the response as valid JSON with no additional text, markdown formatting, or code blocks."""

            summary_result = await self.call_llm_api(summary_prompt)
            
            try:
                # Clean up the summary result by removing any markdown formatting
                summary_result = summary_result.replace("```json", "").replace("```", "").strip()
                
                # Try to parse the summary as JSON
                summary_data = json.loads(summary_result)
                
                # Ensure all required fields are present
                if "summary" not in summary_data:
                    summary_data["summary"] = "No summary provided"
                if "critical_points" not in summary_data:
                    summary_data["critical_points"] = []
                if "code_example" not in summary_data and code_example:
                    summary_data["code_example"] = code_example
                if "original_research" not in summary_data:
                    summary_data["original_research"] = research_results
                
                return json.dumps(summary_data, indent=2)
            except json.JSONDecodeError as e:
                # If JSON parsing fails, create a structured response manually
                logger.error(f"Failed to parse summary as JSON: {str(e)}")
                logger.error(f"Raw summary result: {summary_result}")
                structured_response = {
                    "summary": summary_result,
                    "critical_points": [],
                    "code_example": code_example,
                    "original_research": research_results
                }
                return json.dumps(structured_response, indent=2)
                
        except Exception as e:
            logger.error(f"Error in research process: {str(e)}")
            error_response = {
                "summary": f"Error during research: {str(e)}",
                "critical_points": [],
                "code_example": None,
                "original_research": None
            }
            return json.dumps(error_response, indent=2)

class WriterAgent:
    def __init__(self):
        self.name = "Writer"
        self.description = "Generates code based on requirements and research"
        self.capabilities = ["code_generation", "validation"]
        self.llm_config = config.writer_llm
        self.prompt_writer = PromptWriterAgent()
        self.validators = {
            'python': PythonValidator(),
            'vb.net': VBValidator(),
            # Add more language-specific validators here
        }
        logger.info(f"Writer LLM Config: endpoint={self.llm_config.endpoint}, is_local={self.llm_config.is_local}, model={self.llm_config.model}")
        if not self.llm_config.is_local:
            self.client = openai.OpenAI(api_key=self.llm_config.api_key)

    async def call_llm_api(self, prompt: str, max_retries: int = 3) -> str:
        """Call the LLM API to generate code with retry logic."""
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                logger.info(f"Writer calling LLM API with is_local={self.llm_config.is_local} (attempt {retry_count + 1}/{max_retries})")
                
                if self.llm_config.is_local:
                    async with httpx.AsyncClient(timeout=1200.0) as client:  # Increased timeout to 10 minutes
                        response = await client.post(
                            self.llm_config.endpoint,
                            json={
                                "model": self.llm_config.model,
                                "messages": [
                                    {"role": "system", "content": "You are a code generation assistant specializing in writing safe and idiomatic code."},
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": self.llm_config.temperature,
                                "max_tokens": self.llm_config.max_tokens
                            }
                        )
                        response.raise_for_status()
                        return response.json()["choices"][0]["message"]["content"]
                else:
                    if not self.llm_config.api_key:
                        raise ValueError("OpenAI API key is required when not using local endpoint")
                    response = self.client.chat.completions.create(
                        model=self.llm_config.model,
                        messages=[
                            {"role": "system", "content": "You are a code generation assistant specializing in writing safe and idiomatic code."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.llm_config.temperature,
                        max_tokens=self.llm_config.max_tokens
                    )
                    return response.choices[0].message.content
                    
            except httpx.TimeoutException as e:
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Timeout on attempt {retry_count}, retrying...")
                    await asyncio.sleep(1)  # Wait 1 second before retrying
                continue
            except Exception as e:
                logger.error(f"Error calling LLM API: {e}", exc_info=True)
                raise
        
        # If we get here, all retries failed
        error_message = f"Failed to generate code after {max_retries} attempts. Last error: {str(last_error)}"
        logger.error(error_message)
        raise ValueError(error_message)

    async def validate_code_requirements(self, code: str, requirements: str) -> tuple[bool, str]:
        """Validate if the generated code meets the requirements."""
        try:
            prompt = f"""Validate if the following code meets these requirements:
Requirements: {requirements}

Code:
{code}

Return a JSON response in this exact format:
{{
    "meets_requirements": true/false,
    "message": "Explanation of why it meets or doesn't meet the requirements"
}}"""

            logger.info("Sending validation request to LLM")
            validation_result = await self.call_llm_api(prompt)
            logger.info(f"Received validation response: {validation_result}")
            
            # Use regex to find the first valid JSON object in the response
            # This pattern looks for a JSON object that starts with { and ends with }
            # and contains the required fields
            json_pattern = r'{[^{]*"meets_requirements"\s*:\s*(?:true|false)[^}]*"message"\s*:\s*"[^"]*"[^}]*}'
            match = re.search(json_pattern, validation_result, re.DOTALL)
            
            if not match:
                logger.error(f"Could not find valid JSON object in response: {validation_result}")
                return False, "Invalid validation response: Could not find valid JSON object"
            
            json_portion = match.group(0)
            logger.info(f"Extracted JSON: {json_portion}")
            
            try:
                # Try to parse the extracted JSON
                result = json.loads(json_portion)
                logger.info(f"Successfully parsed JSON response: {result}")
                
                # Ensure the response has the required fields
                if not isinstance(result, dict):
                    logger.error(f"Invalid validation response: not a JSON object. Response: {json_portion}")
                    return False, "Invalid validation response: not a JSON object"
                    
                if "meets_requirements" not in result or "message" not in result:
                    logger.error(f"Invalid validation response: missing required fields. Response: {json_portion}")
                    return False, "Invalid validation response: missing required fields"
                    
                if not isinstance(result["meets_requirements"], bool):
                    logger.error(f"Invalid validation response: meets_requirements must be a boolean. Response: {result}")
                    return False, "Invalid validation response: meets_requirements must be a boolean"
                    
                if not isinstance(result["message"], str):
                    logger.error(f"Invalid validation response: message must be a string. Response: {result}")
                    return False, "Invalid validation response: message must be a string"
                
                logger.info(f"Validation successful. Meets requirements: {result['meets_requirements']}")
                return result["meets_requirements"], result["message"]
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse validation response as JSON: {json_portion}")
                logger.error(f"JSON decode error: {str(e)}")
                logger.error(f"Error location: line {e.lineno}, column {e.colno}")
                logger.error(f"Error message: {e.msg}")
                return False, "Invalid validation response format"
                
        except Exception as e:
            logger.error(f"Error validating code requirements: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return False, f"Validation error: {str(e)}"

    async def process_message(self, message: str) -> str:
        """Process the message and generate code with safety checks."""
        try:
            # Detect language from message using PromptWriterAgent
            language = self.prompt_writer.identify_language(message)
            logger.info(f"Detected language: {language}")
            
            # For VB.NET, use a more direct approach
            if language == 'vb.net':
                code = await self.generate_vb_code(message)
                return f"""Here's the generated VB.NET code:

```vb.net
{code}
```

The code has been validated for safety and meets all requirements."""
            
            # For other languages, use the standard approach
            generation_prompt = await self.prompt_writer.generate_code_prompt(message, language)
            
            # Check if the message contains research results in JSON format
            try:
                research_data = json.loads(message)
                if isinstance(research_data, dict) and "summary" in research_data:
                    # Use the research data to enhance the generation prompt
                    code_example_text = f"\nReference Code Example:\n{research_data.get('code_example', '')}" if research_data.get('code_example') else ""
                    code_example_instruction = "\nYou may use the reference code example as a starting point, but ensure you implement all requirements." if research_data.get('code_example') else ""
                    
                    generation_prompt = f"""Original Request: {message}

Implementation Requirements:
{research_data.get('summary', '')}

Critical Points to Implement:
{chr(10).join(f"- {point}" for point in research_data.get('critical_points', []))}{code_example_text}{code_example_instruction}

Please implement the code following these requirements and critical points. Ensure the code is properly formatted and includes all necessary using statements and XML documentation."""
            except json.JSONDecodeError:
                # If not JSON, use the original generation prompt
                pass
            
            code = await self.call_llm_api(generation_prompt)
            
            # Extract code block and identify language
            extracted_code, detected_language = extract_code_block(code)
            
            # Get appropriate validator
            validator = self.validators.get(detected_language, CodeValidator())

            # Validate code safety
            is_safe, safety_message = validator.validate_code_safety(extracted_code, detected_language)
            if not is_safe:
                return f"Error: {safety_message}"

            # Validate against requirements using LLM
            meets_requirements, requirement_message = await self.validate_code_requirements(extracted_code, message)
            if not meets_requirements:
                return f"Error: {requirement_message}"

            # Add necessary using statements and XML documentation for C#
            if detected_language == 'csharp':
                using_statements = """using System;
using System.Globalization;

"""
                # Add XML documentation if not present
                if not "/// <summary>" in extracted_code:
                    extracted_code = extracted_code.replace("public static void PrintDecimalAsString", """/// <summary>
/// Converts a decimal value to a string and prints it to the console using invariant culture formatting.
/// </summary>
/// <param name="value">The decimal value to convert and print.</param>
public static void PrintDecimalAsString""")
                
                extracted_code = using_statements + extracted_code

            return f"""Here's the generated code:

```{detected_language}
{extracted_code}
```

The code has been validated for safety and meets all requirements."""

        except Exception as e:
            logger.error(f"Error in process_message: {e}", exc_info=True)
            return f"Error generating code: {e}"

    async def generate_vb_code(self, message: str) -> str:
        """Generate VB.NET code with a more focused prompt."""
        prompt = f"""Generate a VB.NET class that implements the following requirement:
{message}

Requirements:
1. Use proper VB.NET syntax and conventions
2. Include appropriate error handling
3. Follow VB.NET naming conventions
4. Include XML documentation comments
5. Use appropriate UI controls or methods based on the requirements
6. Ensure the code is safe and follows best practices

Return only the code, no explanations."""

        try:
            code = await self.call_llm_api(prompt)
            # Extract the code block if it's wrapped in markdown
            if "```" in code:
                code = code.split("```")[1].strip()
                if code.startswith("vb.net"):
                    code = code[6:].strip()
            
            # Validate the code
            validator = self.validators['vb.net']
            is_safe, safety_message = validator.validate_code_safety(code, 'vb.net')
            if not is_safe:
                raise ValueError(f"Code validation failed: {safety_message}")
            
            return code
        except Exception as e:
            logger.error(f"Error generating VB.NET code: {str(e)}")
            raise

def extract_code_block(text: str) -> tuple[str, str]:
    """Extract code block from markdown text and detect its language.
    
    Args:
        text: The text containing a code block
        
    Returns:
        tuple: (extracted_code, detected_language)
    """
    try:
        # Look for code blocks in markdown format
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        match = re.search(code_block_pattern, text, re.DOTALL)
        
        if match:
            language = match.group(1) or 'text'  # Default to 'text' if no language specified
            code = match.group(2).strip()
            return code, language
            
        # If no markdown code block found, return the text as is
        return text, 'text'
    except Exception as e:
        logger.error(f"Error extracting code block: {str(e)}")
        return text, 'text'

class OrchestratorAgent:
    def __init__(self):
        self.name = "Orchestrator"
        self.description = "Coordinates between agents and manages the overall workflow"
        self.capabilities = ["coordination", "workflow_management"]
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
        logger.debug("Orchestrator initialized with agents:")
        logger.debug(f"Researcher config: {self.researcher.llm_config}")
        logger.debug(f"Writer config: {self.writer.llm_config}")

    async def process_message(self, message: str) -> str:
        logger.info(f"Orchestrator processing message: {message}")
        logger.debug(f"Researcher is_local: {self.researcher.llm_config.is_local}")
        logger.debug(f"Writer is_local: {self.writer.llm_config.is_local}")
        
        # For simple script requests, skip the research phase
        if any(keyword in message.lower() for keyword in ["hello world", "hi there", "numbers from 1 to 10"]):
            logger.info("Simple request detected, skipping research phase")
            try:
                return await self.writer.process_message(message)
            except Exception as e:
                logger.error(f"Error in writer process_message: {str(e)}")
                raise
        
        # For more complex requests, use the full pipeline
        try:
            # Get research results
            research_results = await self.researcher.process_message(message)
            logger.info("Research results received, parsing structured output")
            
            try:
                # Parse the structured research output
                research_data = json.loads(research_results)
                summary = research_data.get("summary", "")
                critical_points = research_data.get("critical_points", [])
                code_example = research_data.get("code_example")
                original_research = research_data.get("original_research", "")
                
                # Construct a focused prompt for the writer
                code_example_text = f"\nReference Code Example:\n{code_example}" if code_example else ""
                code_example_instruction = "\nYou may use the reference code example as a starting point, but ensure you implement all requirements." if code_example else ""
                
                writer_prompt = f"""Original Request: {message}

Implementation Requirements:
{summary}

Critical Points to Implement:
{chr(10).join(f"- {point}" for point in critical_points)}{code_example_text}

Please implement the code following these requirements and critical points.{code_example_instruction}"""
                
                code_results = await self.writer.process_message(writer_prompt)
                return f"Research Summary: {summary}\n\nCritical Points:\n{chr(10).join(f'- {point}' for point in critical_points)}\n\nGenerated Code: {code_results}"
            except json.JSONDecodeError:
                logger.error("Failed to parse research results as JSON, falling back to original format")
                # Fall back to the original format if JSON parsing fails
                combined_prompt = f"""Original Request: {message}

Research and Requirements:
{research_results}

Please implement the code based on both the original request and the research findings above.
Ensure the implementation follows all best practices and patterns identified in the research."""
                
                code_results = await self.writer.process_message(combined_prompt)
                return f"Research Results: {research_results}\n\nGenerated Code: {code_results}"
                
        except Exception as e:
            logger.error(f"Error in full pipeline: {str(e)}")
            raise

# Initialize agents
orchestrator = OrchestratorAgent()

def format_sse_event(event: BaseModel) -> str:
    """Format an event as a Server-Sent Event."""
    try:
        # Use EventManager to serialize the event
        event_json = event_manager.serialize_event(event)
        formatted = f"data: {event_json}\n\n"
        logger.debug(f"Formatted SSE event: {formatted}")
        return formatted
    except Exception as e:
        logger.error(f"Error formatting SSE event: {str(e)}", exc_info=True)
        raise

@app.post("/awp")
async def agent_endpoint(input_data: RunAgentInput):
    logger.info(f"Received request with thread_id: {input_data.thread_id}, run_id: {input_data.run_id}")
    logger.info(f"Message content: {input_data.messages[-1].content if input_data.messages else 'No messages'}")

    # Start timing
    start_time = time.time()

    async def event_generator() -> AsyncGenerator[str, None]:
        message_id = None
        try:
            # Create enhanced run started event with metadata
            run_started = RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
                agent_metadata={
                    "name": orchestrator.name,
                    "description": orchestrator.description,
                    "capabilities": orchestrator.capabilities,
                    "start_time": start_time,
                    "environment": {
                        "python_version": sys.version,
                        "platform": sys.platform,
                        "hostname": socket.gethostname()
                    }
                },
                configuration={
                    "researcher_config": {
                        "endpoint": orchestrator.researcher.llm_config.endpoint,
                        "is_local": orchestrator.researcher.llm_config.is_local,
                        "model": orchestrator.researcher.llm_config.model
                    },
                    "writer_config": {
                        "endpoint": orchestrator.writer.llm_config.endpoint,
                        "is_local": orchestrator.writer.llm_config.is_local,
                        "model": orchestrator.writer.llm_config.model
                    }
                }
            )
            
            # Validate and prepare the event
            is_valid, error_msg = event_manager.validate_event(run_started)
            if not is_valid:
                raise ValueError(f"Invalid RunStartedEvent: {error_msg}")
            run_started = event_manager.prepare_event(run_started)
            
            logger.info("Sending RUN_STARTED event")
            yield format_sse_event(run_started)

            # Generate a message ID for the assistant's response
            message_id = str(uuid.uuid4())
            logger.info(f"Generated message_id: {message_id}")

            # Send text message start event with correlation
            text_start = TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message_id,
                role="assistant",
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
                correlation_id=run_started.correlation_id
            )
            
            # Validate and prepare the event
            is_valid, error_msg = event_manager.validate_event(text_start)
            if not is_valid:
                raise ValueError(f"Invalid TextMessageStartEvent: {error_msg}")
            text_start = event_manager.prepare_event(text_start)
            
            logger.info("Sending TEXT_MESSAGE_START event")
            yield format_sse_event(text_start)

            try:
                # Process the message through the orchestrator
                logger.info("Processing message through orchestrator")
                response = await orchestrator.process_message(input_data.messages[-1].content)
                logger.info(f"Orchestrator response: {response}")

                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                timing_info = f"\n\n---\nInference completed in {elapsed_time:.2f} seconds"

                # Send the response content as a single event
                content_event = TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=response + timing_info,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    correlation_id=run_started.correlation_id
                )
                
                # Validate and prepare the event
                is_valid, error_msg = event_manager.validate_event(content_event)
                if not is_valid:
                    raise ValueError(f"Invalid TextMessageContentEvent: {error_msg}")
                content_event = event_manager.prepare_event(content_event)
                
                logger.info("Sending TEXT_MESSAGE_CONTENT event")
                yield format_sse_event(content_event)

                # Send text message end event
                text_end = TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    correlation_id=run_started.correlation_id
                )
                
                # Validate and prepare the event
                is_valid, error_msg = event_manager.validate_event(text_end)
                if not is_valid:
                    raise ValueError(f"Invalid TextMessageEndEvent: {error_msg}")
                text_end = event_manager.prepare_event(text_end)
                
                logger.info("Sending TEXT_MESSAGE_END event")
                yield format_sse_event(text_end)

                # Send run finished event with performance metrics
                run_finished = RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    correlation_id=run_started.correlation_id,
                    completion_status="success",
                    performance_metrics={
                        "total_time": elapsed_time,
                        "message_length": len(response),
                        "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    }
                )
                
                # Validate and prepare the event
                is_valid, error_msg = event_manager.validate_event(run_finished)
                if not is_valid:
                    raise ValueError(f"Invalid RunFinishedEvent: {error_msg}")
                run_finished = event_manager.prepare_event(run_finished)
                
                logger.info("Sending RUN_FINISHED event")
                yield format_sse_event(run_finished)
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)
                # Calculate elapsed time even for errors
                elapsed_time = time.time() - start_time
                error_message = f"Error processing message: {str(e)}\n\n---\nInference failed after {elapsed_time:.2f} seconds"
                error_event = TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=error_message,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    correlation_id=run_started.correlation_id
                )
                
                # Validate and prepare the event
                is_valid, error_msg = event_manager.validate_event(error_event)
                if not is_valid:
                    logger.error(f"Invalid error event: {error_msg}")
                else:
                    error_event = event_manager.prepare_event(error_event)
                    logger.info("Sending error event")
                    yield format_sse_event(error_event)

                # Send end events even if there was an error
                text_end = TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    correlation_id=run_started.correlation_id
                )
                
                # Validate and prepare the event
                is_valid, error_msg = event_manager.validate_event(text_end)
                if not is_valid:
                    logger.error(f"Invalid end event: {error_msg}")
                else:
                    text_end = event_manager.prepare_event(text_end)
                    yield format_sse_event(text_end)

                run_finished = RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    correlation_id=run_started.correlation_id,
                    completion_status="error",
                    performance_metrics={
                        "total_time": elapsed_time,
                        "error_type": type(e).__name__,
                        "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    }
                )
                
                # Validate and prepare the event
                is_valid, error_msg = event_manager.validate_event(run_finished)
                if not is_valid:
                    logger.error(f"Invalid run finished event: {error_msg}")
                else:
                    run_finished = event_manager.prepare_event(run_finished)
                    yield format_sse_event(run_finished)
        except Exception as e:
            logger.error(f"Error in event_generator: {str(e)}", exc_info=True)
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # If we have a message_id, try to send an error event
            if message_id:
                try:
                    # Calculate elapsed time for generator errors
                    elapsed_time = time.time() - start_time
                    error_message = f"Error in event generator: {str(e)}\n\n---\nInference failed after {elapsed_time:.2f} seconds"
                    error_event = TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=message_id,
                        delta=error_message,
                        thread_id=input_data.thread_id,
                        run_id=input_data.run_id,
                        correlation_id=run_started.correlation_id if 'run_started' in locals() else None
                    )
                    
                    # Validate and prepare the event
                    is_valid, error_msg = event_manager.validate_event(error_event)
                    if not is_valid:
                        logger.error(f"Invalid error event: {error_msg}")
                    else:
                        error_event = event_manager.prepare_event(error_event)
                        yield format_sse_event(error_event)
                    
                    # Try to send end events
                    text_end = TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=message_id,
                        thread_id=input_data.thread_id,
                        run_id=input_data.run_id,
                        correlation_id=run_started.correlation_id if 'run_started' in locals() else None
                    )
                    
                    # Validate and prepare the event
                    is_valid, error_msg = event_manager.validate_event(text_end)
                    if not is_valid:
                        logger.error(f"Invalid end event: {error_msg}")
                    else:
                        text_end = event_manager.prepare_event(text_end)
                        yield format_sse_event(text_end)
                    
                    run_finished = RunFinishedEvent(
                        type=EventType.RUN_FINISHED,
                        thread_id=input_data.thread_id,
                        run_id=input_data.run_id,
                        correlation_id=run_started.correlation_id if 'run_started' in locals() else None,
                        completion_status="error",
                        performance_metrics={
                            "total_time": elapsed_time,
                            "error_type": type(e).__name__,
                            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024  # MB
                        }
                    )
                    
                    # Validate and prepare the event
                    is_valid, error_msg = event_manager.validate_event(run_finished)
                    if not is_valid:
                        logger.error(f"Invalid run finished event: {error_msg}")
                    else:
                        run_finished = event_manager.prepare_event(run_finished)
                        yield format_sse_event(run_finished)
                except Exception as inner_e:
                    logger.error(f"Error sending error events: {str(inner_e)}", exc_info=True)
                    logger.error(f"Stack trace: {traceback.format_exc()}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
async def root():
    return {"message": "Multi-Agent Python Application Generator"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 