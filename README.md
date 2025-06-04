# AG-UI Test Application

A FastAPI-based system that demonstrates the implementation of the Agent User Interaction (AG_UI) Protocol for real-time communication between agents and users. This application serves as a practical example of how to integrate AG_UI into an agent-based system.

## Features

- Implementation of AG_UI Protocol for real-time communication
- Server-Sent Events (SSE) for streaming responses
- Structured event system for agent-user interactions
- Real-time code generation with multiple LLM agents
- Support for both cloud-based and local LLM endpoints
- Language-specific code validation and generation
- Comprehensive error handling with AG_UI error events

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Access to an LLM API (OpenAI, local LM Studio, etc.)
- AG_UI Protocol package (`ag_ui_protocol`)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The system uses environment variables for configuration. Create a `.env` file in the project root:

### For Cloud-based LLMs (e.g., OpenAI)

```bash
# Researcher Agent Configuration
RESEARCHER_LLM_ENDPOINT=https://api.openai.com/v1/chat/completions
RESEARCHER_LLM_API_KEY=your-api-key-here
RESEARCHER_LLM_MODEL=gpt-4
RESEARCHER_LLM_TEMPERATURE=0.7
RESEARCHER_LLM_MAX_TOKENS=1000
RESEARCHER_LLM_TIMEOUT=30
RESEARCHER_LLM_IS_LOCAL=false

# Writer Agent Configuration
WRITER_LLM_ENDPOINT=https://api.openai.com/v1/chat/completions
WRITER_LLM_API_KEY=your-api-key-here
WRITER_LLM_MODEL=gpt-4
WRITER_LLM_TEMPERATURE=0.7
WRITER_LLM_MAX_TOKENS=2000
WRITER_LLM_TIMEOUT=30
WRITER_LLM_IS_LOCAL=false

# PromptWriter Agent Configuration
PROMPT_WRITER_LLM_ENDPOINT=https://api.openai.com/v1/chat/completions
PROMPT_WRITER_LLM_API_KEY=your-api-key-here
PROMPT_WRITER_LLM_MODEL=gpt-4
PROMPT_WRITER_LLM_TEMPERATURE=0.7
PROMPT_WRITER_LLM_MAX_TOKENS=1000
PROMPT_WRITER_LLM_TIMEOUT=30
PROMPT_WRITER_LLM_IS_LOCAL=false
```

### For Local LLMs (e.g., LM Studio)

```bash
# Researcher Agent Configuration
RESEARCHER_LLM_ENDPOINT=http://localhost:1234/v1/chat/completions
RESEARCHER_LLM_IS_LOCAL=true
RESEARCHER_LLM_MODEL=local-model-name
RESEARCHER_LLM_TEMPERATURE=0.7
RESEARCHER_LLM_MAX_TOKENS=1000
RESEARCHER_LLM_TIMEOUT=30

# Writer Agent Configuration
WRITER_LLM_ENDPOINT=http://localhost:1234/v1/chat/completions
WRITER_LLM_IS_LOCAL=true
WRITER_LLM_MODEL=local-model-name
WRITER_LLM_TEMPERATURE=0.7
WRITER_LLM_MAX_TOKENS=2000
WRITER_LLM_TIMEOUT=30

# PromptWriter Agent Configuration
PROMPT_WRITER_LLM_ENDPOINT=http://localhost:1234/v1/chat/completions
PROMPT_WRITER_LLM_IS_LOCAL=true
PROMPT_WRITER_LLM_MODEL=local-model-name
PROMPT_WRITER_LLM_TEMPERATURE=0.7
PROMPT_WRITER_LLM_MAX_TOKENS=1000
PROMPT_WRITER_LLM_TIMEOUT=30
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### POST /awp
Main endpoint for agent interactions using AG_UI Protocol.

Request body:
```json
{
    "thread_id": "string",
    "run_id": "string",
    "messages": [
        {
            "content": "string",
            "role": "string"
        }
    ]
}
```

Response: Server-Sent Events (SSE) stream with AG_UI events:
- `RUN_STARTED`
- `TEXT_MESSAGE_START`
- `TEXT_MESSAGE_CONTENT`
- `TEXT_MESSAGE_END`
- `RUN_FINISHED`
- `RUN_ERROR` (if applicable)

## Architecture

The system implements the AG_UI Protocol with the following components:

### Event System
- Uses Server-Sent Events (SSE) for real-time communication
- Implements AG_UI event hierarchy
- Handles event encoding and streaming
- Manages message IDs and event sequencing

### Agent System
- Orchestrator Agent: Coordinates the workflow and event generation
- Researcher Agent: Provides research capabilities
- Writer Agent: Generates and validates code
- PromptWriter Agent: Creates language-specific prompts

### Code Validation
- Language-specific validators
- Safety checks
- Requirement validation
- Module dependency checking

## Error Handling

The system implements comprehensive error handling using AG_UI error events:
- LLM API failures
- Code safety violations
- Missing dependencies
- Invalid configurations
- Language-specific validation errors
- Stream processing errors

## Development

### Adding New Features

1. Update the configuration in `config.py` if needed
2. Modify the agent classes in `main.py`
3. Update the requirements in `requirements.txt`
4. Test with both local and cloud LLM endpoints

### Testing

1. Create a test environment:
```bash
python -m venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate
pip install -r requirements.txt
```

2. Run tests:
```bash
# Add test commands here when tests are implemented
```

## Troubleshooting

### Common Issues

1. LLM API Connection Issues
   - Verify API keys and endpoints
   - Check network connectivity
   - Ensure correct configuration for local/cloud endpoints

2. Event Stream Issues
   - Check SSE connection
   - Verify event formatting
   - Monitor event sequencing

3. Configuration Problems
   - Ensure all required environment variables are set
   - Verify .env file format
   - Check for typos in configuration values

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]

## Support

[Add support contact information here] 