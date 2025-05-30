# Python Code Generation System

A FastAPI-based system that uses LLMs to research and generate Python code based on user requirements.

## Features

- Research best practices and patterns using LLMs
- Generate Python code based on requirements
- Support for both cloud-based and local LLM endpoints
- Configurable LLM settings per agent
- Code safety validation
- Module dependency checking

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Access to an LLM API (OpenAI, local LM Studio, etc.)

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
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### POST /awp
Main endpoint for code generation requests.

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

## Architecture

The system consists of two main agents:

### ResearcherAgent
- Researches best practices and patterns
- Uses LLM to analyze requirements
- Provides structured research results

### WriterAgent
- Generates Python code based on requirements
- Validates code safety
- Checks module dependencies
- Uses LLM for code generation and validation

## Error Handling

The system includes comprehensive error handling for:
- LLM API failures
- Code safety violations
- Missing dependencies
- Invalid configurations

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

2. Code Generation Failures
   - Check LLM model availability
   - Verify token limits
   - Review error logs

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