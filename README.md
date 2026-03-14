# Scalable Agentic AI Framework

An enterprise-grade, production-ready framework for building and scaling Agentic AI workflows. This repository implements modern LLM orchestration patterns with a focus on Go-To-Market (GTM) strategies and multi-step reasoning agents.

## 🚀 Architecture Overview

The framework is built on three core pillars:
1. **Agentic Workflows**: Utilizing multi-step reasoning (Chain-of-Thought) and tool-use to solve complex business problems.
2. **LLM Orchestration**: A robust layer for managing Anthropic (Claude) and AWS Bedrock calls, featuring prompt versioning and evaluation metrics.
3. **Enterprise Governance**: Built-in monitoring with `loguru`, configuration management with `Pydantic`, and multi-stage containerization.

### Core Components
- `src/agents/`: Specialized AI agents with domain-specific reasoning logic.
- `src/core/`: The "Brain" of the framework, handling model routing, fallback logic, and prompt lifecycle.
- `infrastructure/`: Infrastructure-as-code and container orchestration configurations.

## 🛠 Setup Instructions

### Prerequisites
- Python 3.10+
- Docker
- AWS Credentials (for Bedrock) or Anthropic API Key

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/Scalable-Agentic-AI-Framework.git
   cd Scalable-Agentic-AI-Framework
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Running the Service
```bash
docker build -t agent-service -f infrastructure/Dockerfile .
docker run -p 8000:8000 agent-service
```

## 🧠 LLM Orchestration (Anthropic & Bedrock)

This framework abstracts the underlying LLM provider, allowing seamless switching between Anthropic direct APIs and Amazon Bedrock.

- **Claude 3.5 Sonnet**: Used for high-reasoning tasks and complex tool use.
- **Claude 3 Haiku**: Leveraged for low-latency, high-throughput utility tasks.

## 📊 Evaluation & Metrics

The `LLMOrchestrator` includes built-in metrics collection for:
- Token usage and cost estimation.
- Response latency.
- Pass/Fail evaluation based on defined Pydantic schemas.

---
*Maintained by the AI Platform Team.*
