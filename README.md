# Python Code Generation System

A FastAPI-based system that uses LLMs to research and generate code based on user requirements. Supports multiple programming languages including Python, VB.NET, JavaScript, and C#.

## Features

- Research best practices and patterns using LLMs
- Generate code in multiple languages (Python, VB.NET, JavaScript, C#)
- Support for both cloud-based and local LLM endpoints
- Configurable LLM settings per agent
- Language-specific code validation
- Module dependency checking
- Language detection and appropriate prompt generation

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

The system consists of three main agents:

### PromptWriterAgent
- Generates appropriate prompts for different programming languages
- Detects programming language from user input
- Creates language-specific research and code generation prompts
- Ensures consistent prompt structure across languages
- Supports VB.NET, Python, JavaScript, and C# syntax patterns

### ResearcherAgent
- Researches best practices and patterns
- Uses LLM to analyze requirements
- Provides structured research results
- Language-agnostic research capabilities
- Supports multiple programming languages

### WriterAgent
- Generates code based on requirements and research
- Validates code safety using language-specific validators
- Checks module dependencies
- Uses LLM for code generation and validation
- Supports multiple programming languages
- Includes specialized handling for VB.NET code generation

### Code Validation
The system includes a flexible validation framework:
- Base `CodeValidator` class for common validation
- Language-specific validators (e.g., `PythonValidator`, `VBValidator`)
- Extensible design for adding new language validators
- Separate safety checks for different languages
- VB.NET specific validation patterns

## Language Support

### VB.NET
- Full support for VB.NET code generation
- VB.NET specific syntax validation
- MessageBox.Show integration
- XML documentation comments
- Error handling for null/empty values
- Proper VB.NET naming conventions

### Python
- Python code generation
- AST-based syntax validation
- Module dependency checking
- Python-specific safety checks

### JavaScript
- JavaScript code generation
- Syntax validation
- Module dependency checking

### C#
- C# code generation
- Syntax validation
- Module dependency checking

## Error Handling

The system includes comprehensive error handling for:
- LLM API failures
- Code safety violations
- Missing dependencies
- Invalid configurations
- Language-specific validation errors

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