# atk — Agent Toolkit

A Python library that provides a unified, vendor-agnostic interface for building AI agent systems.

![GitHub License](https://img.shields.io/github/license/timurci/atk?style=flat-square)


## Overview

`atk` defines abstract protocols for language models, message schemas, and tool definitions — designed to be importable by anyone building their own agent harness. Vendor integrations (e.g. `atk.openai`) are secondary modules that implement the core protocol.

## Features

- **Core-first design**: `atk.core` defines the `LanguageModel` protocol, message types, and tool schemas — completely free of vendor dependencies
- **Vendor implementations**: Pluggable modules (currently OpenAI) that implement the core protocol
- **Toolset abstraction**: Register callables as tools with automatic parameter extraction from type hints and docstrings
- **Async-native**: The entire `LanguageModel` protocol is async-only
- **Pydantic-based**: All schemas use Pydantic for validation and serialization

## Installation

```bash
uv sync
```

### With OpenAI support

```bash
uv sync --group openai
```

## Usage

```python
from atk.core.toolset import CallableToolset
from atk.openai.model import OpenAILanguageModel

def calculate(operation: str, a: float, b: float) -> str:
    """Perform a mathematical calculation."""
    # ...

toolset = CallableToolset([calculate])

llm = OpenAILanguageModel(base_url="...", api_key="...")
response = await llm.generate_response(
    instruction="You are a helpful assistant.",
    messages=[...],
    tools=toolset.tools,
)
```

## Architecture

```
Vendor implementations (atk.openai, future: atk.huggingface, etc.)
    │  implement
    ▼
Core protocol (atk.core)
    │  defines
    ▼
Pydantic models & types (messages, tools)
```

The core layer is the primary artifact — vendor modules are implementations of its contracts, not peers.

## Project Structure

```
src/atk/
├── core/          # Abstract protocol — vendor-agnostic
└── openai/        # OpenAI SDK implementation

examples/          # Runnable examples (see examples/README.md)
tests/             # Unit tests
```

## Examples

See [examples/README.md](examples/README.md) for runnable demos including:
- Interactive chat loop with streaming responses and tool calling
- Structured output with Pydantic models

## Roadmap

- Language model gateway — instantiate models via string identifiers (e.g. `model="openai:gpt-5.4"`, `model="hf:google/gemma-4-26B-A4B-it:Q4"`) without vendor-specific imports
- Multimodal input support
- HuggingFace Transformers implementation

## Requirements

- Python 3.14+
