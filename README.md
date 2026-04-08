# atk — Agent Toolkit

A Python library that provides a unified, vendor-agnostic interface for building AI agent systems.

![GitHub License](https://img.shields.io/github/license/timurci/atk?style=flat-square)


## Overview

`atk` defines abstract protocols for language models, message schemas, and tool definitions — designed to be importable by anyone building their own agent harness. Provider integrations (e.g. `atk.providers`) are secondary modules that implement the core protocol via the any-llm SDK.

## Features

- **Core-first design**: `atk.core` defines the `LanguageModel` protocol, message types, and tool schemas — completely free of provider dependencies
- **Provider integration**: `atk.providers` uses any-llm to support multiple LLM backends through a single module
- **Thinking/reasoning support**: Native `ThinkingPart` and `ThinkingDelta` types for models that produce reasoning tokens
- **Toolset abstraction**: Register callables as tools with automatic parameter extraction from type hints and docstrings
- **Async-native**: The entire `LanguageModel` protocol is async-only
- **Pydantic-based**: All schemas use Pydantic for validation and serialization

## Installation

```bash
uv sync
```

### With provider support

```bash
uv sync --group providers
```

## Usage

```python
from atk.core.toolset import CallableToolset
from atk.providers.model import AnyLanguageModel

def calculate(operation: str, a: float, b: float) -> str:
    """Perform a mathematical calculation."""
    # ...

toolset = CallableToolset([calculate])

llm = AnyLanguageModel(provider="openai", model="gpt-4o-mini", api_key="...")
response = await llm.generate_response(
    instruction="You are a helpful assistant.",
    messages=[...],
    tools=toolset.tools,
)
```

## Architecture

```
Provider integration (atk.providers via any-llm)
    │  implements
    ▼
Core protocol (atk.core)
    │  defines
    ▼
Pydantic models & types (messages, tools)
```

The core layer is the primary artifact — provider modules are implementations of its contracts, not peers.

## Project Structure

```
src/atk/
├── core/          # Abstract protocol — vendor-agnostic
└── providers/     # any-llm SDK integration

examples/          # Runnable examples (see examples/README.md)
tests/             # Unit tests
```

## Examples

See [examples/README.md](examples/README.md) for runnable demos including:
- Interactive chat loop with streaming responses and tool calling
- Structured output with Pydantic models

## Roadmap

- Multimodal input support

## Requirements

- Python 3.14+
