# Semantic Kernel Recipes

This directory contains recipes for using IBM Granite models with Microsoft's Semantic Kernel framework.

## Recipes

### 1. Minimal Q&A Agent Using Ollama ([Minimal_QA_Agent_Ollama.ipynb](Minimal_QA_Agent_Ollama.ipynb))

A comprehensive guide to creating a minimal Q&A agent using IBM Granite models through Ollama and Semantic Kernel.

**What you'll learn:**
- Setting up Ollama with IBM Granite models
- Configuring Semantic Kernel for local model inference
- Creating interactive chat interfaces
- Managing conversation history
- Implementing conversation summarization
- Customizing model parameters

**Prerequisites:**
- Ollama installed and running
- IBM Granite model pulled (e.g., `granite3-dense:2b`)
- Python environment with Semantic Kernel

**Key Features:**
- ✅ Local inference with Ollama
- ✅ Conversation memory management
- ✅ Interactive chat interface
- ✅ Conversation summarization
- ✅ Configurable model parameters
- ✅ Error handling and troubleshooting

## Getting Started

### Install Ollama

1. Download and install Ollama from [ollama.ai](https://ollama.ai)
2. Pull an IBM Granite model:
   ```bash
   ollama pull granite3-dense:2b
   ```

### Install Python Dependencies

```bash
pip install semantic-kernel aiohttp python-dotenv
```

### Run the Recipes

Open any of the notebook files in Jupyter Lab, VS Code, or your preferred notebook environment and follow along with the instructions.

## About Semantic Kernel

Semantic Kernel is Microsoft's open-source SDK that integrates Large Language Models (LLMs) like IBM Granite with conventional programming languages. It provides:

- **AI orchestration**: Combine AI services with conventional code
- **Plugin system**: Extensible architecture for adding capabilities
- **Memory management**: Built-in conversation and semantic memory
- **Multi-model support**: Works with various AI providers including local models via Ollama

## About Ollama

Ollama is a tool for running large language models locally. It provides:

- **Local inference**: Run models on your own hardware
- **Easy model management**: Simple commands to pull and manage models
- **API compatibility**: OpenAI-compatible API endpoints
- **Resource optimization**: Efficient model loading and memory management

## IBM Granite Models

IBM Granite models are a family of decoder-only large language models designed for:

- **Code generation**: Programming and software development tasks
- **Instruction following**: Following complex instructions accurately
- **Reasoning**: Multi-step reasoning and problem-solving
- **Safety**: Built-in safety measures and responsible AI practices

Available models in Ollama:
- `granite3-dense:2b` - Lightweight model for quick responses
- `granite3-dense:8b` - Balanced performance and capability
- `granite3-moe:3b` - Mixture of experts architecture

## Additional Resources

- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Ollama Documentation](https://ollama.ai/docs)
- [IBM Granite Models on Ollama](https://ollama.com/blog/ibm-granite)
- [IBM Granite GitHub Repository](https://github.com/ibm-granite)
