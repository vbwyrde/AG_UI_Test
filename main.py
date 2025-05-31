from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from ag_ui.core import (
    RunAgentInput,
    Message,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
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
from typing import List, Optional, AsyncGenerator
from pydantic import BaseModel
import ast
import re
import importlib
import openai
import httpx
from dotenv import load_dotenv
from config import config

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

class ResearcherAgent:
    def __init__(self):
        self.name = "Researcher"
        self.description = "Researches best practices and patterns for Python applications"
        self.capabilities = ["research", "analysis"]
        self.llm_config = config.researcher_llm
        logger.info(f"Researcher LLM Config: endpoint={self.llm_config.endpoint}, is_local={self.llm_config.is_local}, model={self.llm_config.model}")
        # Only initialize OpenAI client if not using local endpoint
        if not self.llm_config.is_local:
            self.client = openai.OpenAI(api_key=self.llm_config.api_key)

    async def call_llm_api(self, prompt: str) -> str:
        """Call the LLM API to perform research."""
        try:
            logger.info(f"Researcher calling LLM API with is_local={self.llm_config.is_local}")
            logger.info(f"Researcher endpoint: {self.llm_config.endpoint}")
            logger.info(f"Researcher model: {self.llm_config.model}")
            
            if self.llm_config.is_local:
                # Use httpx for local endpoints
                logger.info(f"Using local endpoint: {self.llm_config.endpoint}")
                async with httpx.AsyncClient(timeout=self.llm_config.timeout) as client:
                    response = await client.post(
                        self.llm_config.endpoint,
                        json={
                            "model": self.llm_config.model,
                            "messages": [
                                {"role": "system", "content": "You are a Python code research assistant. Provide detailed, structured research about Python best practices and patterns."},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": self.llm_config.temperature,
                            "max_tokens": self.llm_config.max_tokens
                        }
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
            else:
                # Use OpenAI client for cloud endpoints
                logger.info("Using OpenAI endpoint")
                if not self.llm_config.api_key:
                    raise ValueError("OpenAI API key is required when not using local endpoint")
                response = self.client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=[
                        {"role": "system", "content": "You are a Python code research assistant. Provide detailed, structured research about Python best practices and patterns."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            raise

    async def process_message(self, message: str) -> str:
        logger.info(f"Researcher processing message: {message}")
        
        # Create a research prompt for the LLM
        research_prompt = f"""
        Please research and provide best practices and patterns for implementing the following Python code requirement:
        {message}
        
        Focus on:
        1. Best practices for this type of code
        2. Common patterns that could be applied
        3. Potential pitfalls to avoid
        4. Recommended approaches
        
        Return the results in a structured format.
        """
        
        try:
            # Get research results from LLM
            research_results = await self.call_llm_api(research_prompt)
            return research_results
        except Exception as e:
            logger.error(f"Error in research process: {str(e)}")
            return f"Error during research: {str(e)}"

def extract_code_block(generated_code: str) -> tuple[str, str]:
    """Extracts code block and identifies language from generated code."""
    # Default values
    code_block = generated_code
    language = "python"  # Default to Python
    
    # Find code block markers
    start_marker_pattern = r"```(\w+)?"
    start_match = re.search(start_marker_pattern, generated_code)
    
    if start_match:
        start_marker = start_match.group()
        start_index = start_match.start()
        
        # Find the end of the code block
        end_marker = start_marker[:3]
        end_index = generated_code.find(end_marker, start_index + len(start_marker))
        
        if end_index != -1:
            code_block = generated_code[start_index + len(start_marker):end_index].strip()
            
            # Determine the language
            language_match = re.search(r"(\w+)", start_marker)
            if language_match:
                language = language_match.group(1).lower()
    
    return code_block, language

def identify_language(code: str) -> str:
    """Identifies the programming language of the code."""
    # Simple language detection based on syntax patterns
    patterns = {
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
    
    # Count matches for each language
    scores = {lang: 0 for lang in patterns}
    for lang, lang_patterns in patterns.items():
        for pattern in lang_patterns:
            if re.search(pattern, code):
                scores[lang] += 1
    
    # Return the language with the highest score, default to Python
    return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else 'python'

def check_module_dependencies(code: str) -> tuple[bool, List[str]]:
    """Checks if all required modules are available."""
    try:
        # Extract import statements
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import',
            r'import\s+(\w+)\s+as'
        ]
        
        required_modules = set()
        for pattern in import_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                module_name = match.group(1)
                required_modules.add(module_name)
        
        # Check if modules are available
        missing_modules = []
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                missing_modules.append(module)
        
        return len(missing_modules) == 0, missing_modules
    
    except Exception as e:
        logger.error(f"Error checking module dependencies: {str(e)}")
        return False, [str(e)]

class WriterAgent:
    def __init__(self):
        self.name = "Writer"
        self.description = "Generates Python code based on requirements and research"
        self.capabilities = ["code_generation", "validation"]
        self.llm_config = config.writer_llm
        logger.info(f"Writer LLM Config: endpoint={self.llm_config.endpoint}, is_local={self.llm_config.is_local}, model={self.llm_config.model}")
        logger.debug(f"Full Writer LLM Config: {self.llm_config}")
        # Only initialize OpenAI client if not using local endpoint
        if not self.llm_config.is_local:
            logger.info("Initializing OpenAI client for Writer")
            self.client = openai.OpenAI(api_key=self.llm_config.api_key)
        else:
            logger.info("Writer configured to use local endpoint")

    async def call_llm_api(self, prompt: str) -> str:
        """Call the LLM API to generate code."""
        try:
            logger.info(f"Writer calling LLM API with is_local={self.llm_config.is_local}")
            logger.info(f"Writer endpoint: {self.llm_config.endpoint}")
            logger.info(f"Writer model: {self.llm_config.model}")
            
            if self.llm_config.is_local:
                # Use httpx for local endpoints
                logger.info(f"Using local endpoint: {self.llm_config.endpoint}")
                # Set longer timeouts for local LLM
                timeout = httpx.Timeout(300.0, connect=10.0, read=600.0, write=10.0)  # 10 minute timeout for response
                async with httpx.AsyncClient(
                    timeout=timeout,
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                    transport=httpx.AsyncHTTPTransport(retries=3)
                ) as client:
                    # Create a more specific prompt with example
                    system_prompt = """You are a Python code generation assistant. Generate only the code, no explanations.
                    The code should:
                    1. Be safe to run
                    2. Follow Python best practices
                    3. Include proper error handling
                    4. Be well-documented
                    5. Include a main guard
                    
                    Example format for printing numbers:
                    ```python
                    def main():
                        try:
                            # Print numbers from 1 to 5
                            for i in range(1, 6):
                                print(i)
                            return 0
                        except Exception as e:
                            print(f"Error: {e}")
                            return 1

                    if __name__ == "__main__":
                        exit(main())
                    ```
                    
                    Always use this exact structure, just modify the code inside the try block."""
                    
                    request_data = {
                        "model": self.llm_config.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Create a Python script that: {prompt}"}
                        ],
                        "temperature": 0.3,  # Lower temperature for more deterministic output
                        "max_tokens": 500  # Reduce max tokens for faster response
                    }
                    logger.debug(f"Local API request data: {request_data}")
                    logger.info("Sending request to local LLM...")
                    try:
                        response = await client.post(
                            self.llm_config.endpoint,
                            json=request_data,
                            timeout=timeout
                        )
                        logger.info(f"Received response from local LLM with status: {response.status_code}")
                        response.raise_for_status()
                        response_data = response.json()
                        logger.debug(f"Local LLM response data: {response_data}")
                        
                        # Extract the response content
                        if "choices" in response_data and len(response_data["choices"]) > 0:
                            message = response_data["choices"][0].get("message", {})
                            content = message.get("content", "")
                            logger.info(f"Extracted content from response: {content[:100]}...")  # Log first 100 chars
                            return content
                        else:
                            logger.error(f"Unexpected response format: {response_data}")
                            raise ValueError("Unexpected response format from local LLM")
                    except httpx.TimeoutException:
                        logger.error("Request to local LLM timed out")
                        raise ValueError("Local LLM request timed out. The model is taking longer than expected to respond. Please try again.")
                    except httpx.RequestError as e:
                        logger.error(f"Request to local LLM failed: {str(e)}")
                        raise ValueError(f"Failed to connect to local LLM: {str(e)}")
            else:
                # Use OpenAI client for cloud endpoints
                logger.info("Using OpenAI endpoint")
                if not self.llm_config.api_key:
                    raise ValueError("OpenAI API key is required when not using local endpoint")
                response = self.client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Create a Python script that: {prompt}"}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            raise

    def validate_code_safety(self, code: str) -> tuple[bool, str]:
        """Validates if the code is safe to run."""
        try:
            # Basic syntax validation
            ast.parse(code)
            
            # Check for dangerous operations
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
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    async def validate_code_requirements(self, code: str, requirements: str) -> tuple[bool, str]:
        """Validates if the code meets the specified requirements using LLM."""
        try:
            # Create a validation prompt for the LLM
            validation_prompt = f"""
            Please validate if the following Python code meets these requirements:
            
            Requirements:
            {requirements}
            
            Code to validate:
            ```python
            {code}
            ```
            
            Analyze if the code:
            1. Fulfills all requirements
            2. Follows best practices
            3. Is properly structured
            4. Has appropriate error handling
            
            Return a validation result with explanation.
            """
            
            # Get validation from LLM
            validation_result = await self.call_llm_api(validation_prompt)
            return True, validation_result
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    async def process_message(self, message: str) -> str:
        """Process the message and generate code with safety checks."""
        try:
            # Create a code generation prompt for the LLM
            generation_prompt = f"""
            Please generate a Python script that meets the following requirements:
            {message}
            
            The code should:
            1. Be safe to run
            2. Follow Python best practices
            3. Include proper error handling
            4. Be well-documented
            5. Include a main guard
            
            Return only the Python code, no explanations.
            """
            
            # Generate code using LLM
            code = await self.call_llm_api(generation_prompt)
            
            # Extract code block and identify language
            extracted_code, detected_language = extract_code_block(code)
            identified_language = identify_language(extracted_code)
            
            # Log language detection
            logger.info(f"Detected language: {detected_language}, Identified language: {identified_language}")

            # Check module dependencies
            if identified_language == 'python':
                modules_available, missing_modules = check_module_dependencies(extracted_code)
                if not modules_available:
                    return f"Error: Missing required modules: {', '.join(missing_modules)}"

            # Validate code safety
            is_safe, safety_message = self.validate_code_safety(extracted_code)
            if not is_safe:
                return f"Error: {safety_message}"

            # Validate against requirements using LLM
            meets_requirements, requirement_message = await self.validate_code_requirements(extracted_code, message)
            if not meets_requirements:
                return f"Error: {requirement_message}"

            # Format the response with markdown
            return f"""Here's the generated code:

```{identified_language}
{extracted_code}
```

The code has been validated for safety and meets all requirements."""

        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}")
            return f"Error generating code: {str(e)}"

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
            research_results = await self.researcher.process_message(message)
            code_results = await self.writer.process_message(research_results)
            return f"Research Results: {research_results}\n\nGenerated Code: {code_results}"
        except Exception as e:
            logger.error(f"Error in full pipeline: {str(e)}")
            raise

# Initialize agents
orchestrator = OrchestratorAgent()

def format_sse_event(event: BaseModel) -> str:
    """Format an event as a Server-Sent Event."""
    try:
        event_json = event.model_dump_json()
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

    async def event_generator() -> AsyncGenerator[str, None]:
        message_id = None
        try:
            # Send run started event
            run_started = RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id
            )
            logger.info("Sending RUN_STARTED event")
            yield format_sse_event(run_started)

            # Generate a message ID for the assistant's response
            message_id = str(uuid.uuid4())
            logger.info(f"Generated message_id: {message_id}")

            # Send text message start event
            text_start = TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message_id,
                role="assistant"
            )
            logger.info("Sending TEXT_MESSAGE_START event")
            yield format_sse_event(text_start)

            try:
                # Process the message through the orchestrator
                logger.info("Processing message through orchestrator")
                response = await orchestrator.process_message(input_data.messages[-1].content)
                logger.info(f"Orchestrator response: {response}")

                # Send the response content as a single event
                content_event = TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=response
                )
                logger.info("Sending TEXT_MESSAGE_CONTENT event")
                yield format_sse_event(content_event)

                # Send text message end event
                text_end = TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id
                )
                logger.info("Sending TEXT_MESSAGE_END event")
                yield format_sse_event(text_end)

                # Send run finished event
                run_finished = RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id
                )
                logger.info("Sending RUN_FINISHED event")
                yield format_sse_event(run_finished)
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)
                error_event = TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=f"Error processing message: {str(e)}"
                )
                logger.info("Sending error event")
                yield format_sse_event(error_event)

                # Send end events even if there was an error
                text_end = TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id
                )
                yield format_sse_event(text_end)

                run_finished = RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id
                )
                yield format_sse_event(run_finished)
        except Exception as e:
            logger.error(f"Error in event_generator: {str(e)}", exc_info=True)
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # If we have a message_id, try to send an error event
            if message_id:
                try:
                    error_event = TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=message_id,
                        delta=f"Error in event generator: {str(e)}"
                    )
                    yield format_sse_event(error_event)
                    
                    # Try to send end events
                    text_end = TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=message_id
                    )
                    yield format_sse_event(text_end)
                    
                    run_finished = RunFinishedEvent(
                        type=EventType.RUN_FINISHED,
                        thread_id=input_data.thread_id,
                        run_id=input_data.run_id
                    )
                    yield format_sse_event(run_finished)
                except Exception as inner_e:
                    logger.error(f"Error sending error events: {str(inner_e)}", exc_info=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
            "Transfer-Encoding": "chunked"
        }
    )

@app.get("/")
async def root():
    return {"message": "Multi-Agent Python Application Generator"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 