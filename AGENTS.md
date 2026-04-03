# AGENTS.md — atk

<!--
  This file is read by AI coding agents at the start of every session.
  Keep it accurate, imperative, and free of speculation.
-->

## Project Overview

`atk` (Agent Toolkit) is a Python library that provides a unified, vendor-agnostic interface for building AI agent systems. The core package (`atk.core`) defines abstract protocols for language models, message schemas, and tool definitions — designed to be importable by anyone building their own agent harness. Vendor integrations (e.g. `atk.openai`) are secondary modules that implement the core protocol. The project is structured as an extensible toolkit, with planned future modules for tools (`atk.tools`) and a proper agent harness.

---

## Tech Stack

- **Language**: Python `3.14+`
- **Package manager / runner**: `uv`
- **Dependencies declared in**: `pyproject.toml`
- **Linter + formatter**: `ruff` (config in `ruff.toml`)
- **Static type checker**: `ty`
- **Pre-commit hooks**: `prek` (config in `prek.toml`)
- **Testing**: `pytest`
- **Key libraries**:
  - `pydantic` — data models and validation (core dependency)
  - `openai` — OpenAI SDK (optional, in `[dependency-groups.openai]`)

---

## Environment

### Install dependencies

```bash
uv sync
```

### Install with optional OpenAI support

```bash
uv sync --group openai
```

### Run tests

```bash
uv run pytest
```

---

## Quality Gates

**Every change must pass all of the following before it is considered complete:**

```bash
# 1. Fix lint issues
uv run ruff check --fix

# 2. Format code
uv run ruff format

# 3. Type check
uv run ty check

# 4. Run tests
uv run pytest
```

Run these in order. Fix all errors and warnings before finishing. Do not submit a change that fails any of these.

---

## Pre-commit Hooks

Install once after cloning:

```bash
prek install
```

Run manually against all files:

```bash
prek run --all-files
```

Hooks enforce: conventional commit messages, ruff check/format, ty type checking, and safety checks (private keys, large files, whitespace, TOML/YAML/JSON validity).

---

## Project Structure

```
atk/
├── pyproject.toml              # Project metadata and dependencies
├── ruff.toml                   # Ruff linter/formatter config
├── prek.toml                   # Pre-commit hook configuration
├── AGENTS.md                   # This file
│
├── src/
│   └── atk/
│       ├── __init__.py
│       ├── core/               # Abstract protocol — vendor-agnostic
│       │   ├── __init__.py
│       │   ├── model.py        # LanguageModel protocol (async interface)
│       │   ├── message.py      # Message schemas (User/Assistant/Tool messages)
│       │   └── tool.py         # Tool schema and parameter types
│       └── openai/             # OpenAI SDK implementation
│           ├── __init__.py
│           ├── model.py        # OpenAILanguageModel implements LanguageModel
│           ├── message.py      # OpenAIMessageMapper — bidirectional mapping
│           └── tool.py         # OpenAIToolMapper — internal → OpenAI schema
│
├── tests/
│   ├── __init__.py
│   └── unit/
│       ├── __init__.py
│       └── core/
│           ├── __init__.py
│           ├── conftest.py     # Shared fixtures
│           ├── test_tool_from_callable.py
│           └── tool.py         # Test fixtures / helpers
│
├── examples/
│   ├── __init__.py
│   └── openai/                 # OpenAI usage examples
│
└── docs/                       # Documentation
```

---

## Architecture & Design

`atk` has two explicit layers. Maintain this boundary strictly:

```
Vendor implementations (atk.openai, future: atk.anthropic, etc.)
    │  implement
    ▼
Core protocol (atk.core)
    │  defines
    ▼
Pydantic models & types (message.py, tool.py)
```

- **Core layer** (`atk.core/`): Defines the `LanguageModel` protocol, message types (`UserMessage`, `AssistantMessage`, `ToolMessage`), and tool parameter schemas. This is the library surface — anyone should be able to import `atk.core` and build their own agent system without any vendor dependency. Keep it clean, abstract, and free of vendor-specific code.
- **Vendor layer** (`atk.openai/` and future modules): Implements the core protocol for a specific provider. Contains mappers that translate between internal types and vendor SDK types. Each vendor module is independent — do not share state or imports between vendor modules except through `atk.core`.

**Design principles:**
- **YAGNI**: Do not add plugin systems, registries, or abstract base classes unless a concrete second implementation exists or is explicitly required.
- **KISS**: Prefer plain functions and Pydantic models over class hierarchies. Introduce abstractions only when duplication is clear and justified.
- **Core-first**: `atk.core` is the primary artifact. Vendor modules are implementations of its contracts, not peers.

---

## Code Conventions

### Naming
- Modules and packages: `snake_case`
- Classes: `PascalCase`
- Functions, variables, parameters: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

### Typing
- All public function signatures must have complete type annotations.
- Prefer `X | None` over `Optional[X]`.
- Do not use `Any` unless interfacing with an untyped third-party library, and always add a comment explaining why.

### Error handling
- Raise `NotImplementedError` with a descriptive message for unimplemented methods or unsupported cases.
- Never raise bare `Exception` or `ValueError` — use specific exception types or `NotImplementedError`.
- Vendor modules should raise `NotImplementedError` for unsupported features (e.g. audio output, custom tool calls) with a clear message.

### Imports
- Import order: stdlib → third-party → internal (ruff enforces this).
- Never use wildcard imports.
- Vendor modules must import from `atk.core`, never from other vendor modules.

### Async design
- The `LanguageModel` protocol is async-only (`async def generate_response`). All vendor implementations must follow this pattern. Do not add synchronous implementations.

---

## Testing Conventions

- Test files live in `tests/unit/` and mirror the `src/atk/` structure.
- Use `conftest.py` for shared fixtures — do not hardcode test data inline across multiple tests.
- Tests should be independent and not rely on external state or execution order.
- Every new public function or class must have at least one test.
- Parametrize edge cases explicitly (empty inputs, unsupported types, error conditions).

---

## What NOT to Do

- **Do not** add dependencies without `uv add [package]` (updates `pyproject.toml` and lockfile).
- **Do not** use `pip install` directly.
- **Do not** put vendor-specific code in `atk.core/` — it must remain provider-agnostic.
- **Do not** import from one vendor module into another (e.g. `atk.openai` must not import from a future `atk.anthropic`).
- **Do not** add synchronous implementations of `LanguageModel` — the protocol is async-only.
- **Do not** suppress type errors with `# type: ignore` without an explanatory comment.
- **Do not** edit `ruff.toml` or add `# noqa` suppressions without a comment explaining the exception.
- **Do not** introduce plugin systems or registries unless a concrete second vendor implementation exists.

---

## Known Gotchas

- Python `3.14+` is required — do not use syntax or features unavailable in this version.
- The `openai` dependency is optional (in `[dependency-groups.openai]`). Code in `atk.openai/` will fail to import without it — this is intentional.
- Ruff is configured with `select = ["ALL"]` and only ignores `COM812` globally. Per-file ignores for `tests/**` disable annotation requirements, assert bans, and docstring requirements.
