# Multi-Agent Python Application Generator

This demonstration project showcases a multi-agent system using AG-UI and MCP (Model Context Protocol) to create Python applications through natural language interaction.

## Components

- **Researcher Agent**: Researches best practices and patterns for Python applications
- **Writer Agent**: Generates code based on research and requirements
- **Orchestrator Agent**: Coordinates between agents and manages the overall workflow

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage

1. Start the application
2. Open your web browser to http://localhost:8000
3. Begin chatting with the agents to create your Python application

## Architecture

The system uses AG-UI for the frontend interface and MCP for agent communication. The agents work together to:
1. Understand user requirements
2. Research appropriate solutions
3. Generate and validate code
4. Provide feedback and suggestions 