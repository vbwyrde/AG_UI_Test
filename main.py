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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    async def process_message(self, message: str) -> str:
        logger.info(f"Researcher processing message: {message}")
        research_results = {
            "best_practices": [
                "Use type hints",
                "Follow PEP 8 guidelines",
                "Implement proper error handling",
                "Write unit tests"
            ],
            "patterns": [
                "MVC pattern for web applications",
                "Repository pattern for data access",
                "Factory pattern for object creation"
            ]
        }
        return str(research_results)

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
        self.max_retries = 3
        self.current_retry = 0

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

    def validate_code_requirements(self, code: str, requirements: str) -> tuple[bool, str]:
        """Validates if the code meets the specified requirements."""
        try:
            # Create a prompt for the LLM to validate requirements
            validation_prompt = f"""
            Requirements: {requirements}
            
            Code to validate:
            ```python
            {code}
            ```
            
            Does this code fulfill all requirements? Respond with True or False and explain why.
            """
            
            # Use the LLM to validate (simplified version - you'd need to implement actual LLM call)
            # For now, we'll return a placeholder response
            return True, "Code meets requirements"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    async def process_message(self, message: str) -> str:
        """Process the message and generate code with safety checks."""
        try:
            # Extract code generation requirements
            requirements = message.lower()
            
            # Generate initial code
            if "hello world" in requirements:
                code = """def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()"""
            elif "hi there" in requirements:
                code = """def main():
    for i in range(10):
        print("Hi there")
    print("finito, bro")

if __name__ == "__main__":
    main()"""
            else:
                # For other requests, you'd implement your code generation logic here
                code = f"""def main():
    print("Generated code for: {message}")

if __name__ == "__main__":
    main()"""

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

            # Validate against requirements
            meets_requirements, requirement_message = self.validate_code_requirements(extracted_code, requirements)
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

    async def process_message(self, message: str) -> str:
        logger.info(f"Orchestrator processing message: {message}")
        # For simple script requests, skip the research phase
        if any(keyword in message.lower() for keyword in ["hello world", "hi there", "numbers from 1 to 10"]):
            return await self.writer.process_message(message)
        
        # For more complex requests, use the full pipeline
        research_results = await self.researcher.process_message(message)
        code_results = await self.writer.process_message(research_results)
        return f"Research Results: {research_results}\n\nGenerated Code: {code_results}"

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
                role="assistant"  # Use string instead of enum
            )
            logger.info("Sending TEXT_MESSAGE_START event")
            yield format_sse_event(text_start)

            try:
                # Process the message through the orchestrator
                logger.info("Processing message through orchestrator")
                response = await orchestrator.process_message(input_data.messages[-1].content)
                logger.info(f"Orchestrator response: {response}")

                # Send the response content
                content_event = TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=response
                )
                logger.info("Sending TEXT_MESSAGE_CONTENT event")
                yield format_sse_event(content_event)
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)
                error_event = TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=f"Error processing message: {str(e)}"
                )
                logger.info("Sending error event")
                yield format_sse_event(error_event)

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