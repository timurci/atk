# AGENTS.md — atk

<!--
  This file is read by AI coding agents at the start of every session.
  Keep it accurate, imperative, and free of speculation.
-->

## Project Overview

`atk` (Agent Toolkit) is a Python library that provides a unified, vendor-agnostic interface for building AI agent systems. The core package (`atk.core`) defines abstract protocols for language models, message schemas, and tool definitions — designed to be importable by anyone building their own agent harness. Provider integrations (e.g. `atk.providers`) are secondary modules that implement the core protocol using third-party SDKs. The project is structured as an extensible toolkit, with planned future modules for tools (`atk.tools`) and a proper agent harness.

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
  - `any-llm-sdk` — unified LLM provider interface (optional)

---

## Environment

### Install dependencies

```bash
uv sync
```

### Install with optional provider support

```bash
uv sync --extra providers
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
│       │   ├── tool.py         # Tool schema and parameter types
│       │   └── toolset.py      # CallableToolset — invoke tools from Tool definitions
│       └── providers/          # any-llm integration (3rd-party adapter)
│           ├── __init__.py
│           ├── model.py        # AnyLanguageModel implements LanguageModel
│           ├── message.py      # MessageMapper — bidirectional mapping
│           └── tool.py         # ToolMapper — internal → any-llm tool schema
│
├── tests/
│   ├── __init__.py
│   └── unit/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── conftest.py     # Shared fixtures
│       │   ├── test_tool_from_callable.py
│       │   └── tool.py         # Test fixtures / helpers
│       └── providers/
│           ├── __init__.py
│           ├── test_message_mapper.py
│           ├── test_model.py
│           ├── test_stream_accumulator.py
│           └── test_tool_mapper.py
│
├── examples/
│   ├── __init__.py
│   ├── structured.py           # Structured output example
│   └── chat/                   # Interactive chat example
│       ├── __init__.py
│       ├── main.py
│       ├── chat_loop.py
│       ├── display.py
│       └── tools.py
│
└── docs/                       # Documentation
```

---

## Architecture & Design

`atk` has two explicit layers. Maintain this boundary strictly:

```
Provider integrations (atk.providers)
    │  implement
    ▼
Core protocol (atk.core)
    │  defines
    ▼
Pydantic models & types (message.py, tool.py)
```

- **Core layer** (`atk.core/`): Defines the `LanguageModel` protocol, message types (`UserMessage`, `AssistantMessage`, `ToolMessage`), and tool parameter schemas. This is the library surface — anyone should be able to import `atk.core` and build their own agent system without any provider dependency. Keep it clean, abstract, and free of provider-specific code.
- **Provider layer** (`atk.providers/`): A **third-party integration adapter** that implements the core protocol using the `any-llm-sdk` library. Contains mappers that translate between internal types and any-llm SDK types. This module is independent and should not be imported by other vendor modules except through `atk.core`. Because it depends on `any-llm-sdk`, it delegates provider selection to any-llm (which uses official provider SDKs under the hood).

**Design principles:**
- **KISS**: Prefer the simplest implementation that satisfies the current requirement. Prefer plain functions and Pydantic models over class hierarchies. Avoid premature abstraction, unnecessary indirection, and abstractions that do not remove real duplication.
- **YAGNI**: Do not add plugin systems, registries, config hooks, extensibility points, or abstract base classes unless a concrete second implementation exists or is explicitly required.
- **Core-first**: `atk.core` is the primary artifact. Provider modules are implementations of its contracts, not peers.
- **Let errors propagate**: Do not catch exceptions only to return a silent fallback (e.g., empty bytes, `None`, or an empty list). That masks the real failure and makes debugging harder. Only catch exceptions if the code can meaningfully recover or add context; otherwise, let the exception propagate.

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
- Do not use `cast`. Use type narrowing, validation, overloads, or clearer data modeling instead.

### Error handling
- Raise `NotImplementedError` with a descriptive message for unimplemented methods or unsupported cases.
- Never raise bare `Exception` or `ValueError` — use specific exception types or `NotImplementedError`.
- Custom project error classes must inherit directly from `Exception`, not from built-in exception subclasses.
- Provider modules should raise `NotImplementedError` for unsupported features (e.g. audio output, custom tool calls) with a clear message.

### Imports
- Import order: stdlib → third-party → internal (ruff enforces this).
- Never use wildcard imports.

### Async design
- The `LanguageModel` protocol is async-only (`async def generate_response`). All provider implementations must follow this pattern. Do not add synchronous implementations.

---

## Testing Conventions

- Unit tests live in `tests/unit/` and mirror the `src/atk/` structure. These tests exercise individual functions and classes in isolation — no network calls, no external services.
- Use `conftest.py` for shared fixtures — do not hardcode test data inline across multiple tests.
- Tests should be independent and not rely on external state or execution order.

### When to add tests

- **New public function or class**: Add at least one test covering the primary happy path and any documented edge cases.
- **Bug fix**: Add a test that reproduces the bug (it must fail without the fix) and passes with the fix in place. This prevents regressions.
- **New behavior or feature**: Add tests that exercise the documented contract — both the expected result and the error/edge cases the feature explicitly handles.

Not every change requires a test. Use judgment:
- Pure refactors that preserve behavior (renames, extracting functions, reordering) do not need new tests.
- Changes to private helpers with no new branching logic may not need tests if existing coverage already exercises them.
- Documentation, config, or formatting changes never need tests.

### Avoiding redundant or low-value tests

A test is **low-value** when removing it does not reduce confidence in correctness. Apply these rules:

- **Do not duplicate coverage**: If test A and test B exercise the exact same code path with the same assertions, keep only one. Use `@pytest.mark.parametrize` to consolidate inputs that differ only in data, not logic.
- **Do not test the language or stdlib**: Asserting that `str(42) == "42"` or that `json.loads("{}")` returns `{}` tests Python, not your code. Test the code path that *produces* or *consumes* the value, not the built-in conversion.
- **Do not test implementation details that Pydantic already guarantees**: For example, verifying that a Pydantic model's fields all appear after construction is redundant if the model validates on instantiation.
- **Do not parametrize over trivially different inputs when the mapping logic is already proven**: Once you have tested that `_map_type(str, ...)` returns a `PrimitiveParameter(type="string")`, you do not need a separate test for `_map_type(float, ...)` — unless the mapping differs per type. If all cases follow the same pattern, parametrize them together in a single test rather than scattering identical assertions across multiple methods.
- **Every test must protect against a distinct failure mode**: Before writing a test, ask "what real bug would this catch?" If the answer is "none" — e.g., the assertion can only fail due to a change in the same test — the test is low-value.

### Test organization

- Group tests by class or concern (e.g. `TestToolMetadata`, `TestEdgeCases`) so that related assertions are co-located.
- Parametrize data-driven inputs rather than copy-pasting test methods with different values.
- Keep test fixtures in `conftest.py` when reused across test files. Prefer inline definitions for single-use helpers.
- Use `@pytest.mark.parametrize` for edge cases (empty inputs, unsupported types, error conditions) rather than writing separate test methods for each.

---

## What NOT to Do

- **Do not** add dependencies without `uv add [package]` (updates `pyproject.toml` and lockfile).
- **Do not** use `pip install` directly.
- **Do not** put provider-specific code in `atk.core/` — it must remain provider-agnostic.
- **Do not** import from one provider module into another.
- **Do not** add synchronous implementations of `LanguageModel` — the protocol is async-only.
- **Do not** suppress type errors with `# type: ignore` without an explanatory comment.
- **Do not** use `cast`; model or narrow the type properly.
- **Do not** edit `ruff.toml` or add `# noqa` suppressions without a comment explaining the exception.
- **Do not** introduce plugin systems or registries unless a concrete second provider implementation exists.
- **Do not** write tests that exercise the same code path with identical assertions — consolidate with `@pytest.mark.parametrize` or merge into the broader test.
- **Do not** test Python built-in behavior (e.g., `str()` conversion, `json.loads`) — test your code that calls them.
- **Do not** add a test without asking whether a real bug would cause it to fail. If the test can only fail due to an error in the test itself, remove it.

---

## Known Gotchas

- Python `3.14+` is required — do not use syntax or features unavailable in this version.
- The `any-llm-sdk` dependency is optional. Code in `atk.providers/` will fail to import without it — this is intentional.
- Ruff is configured with `select = ["ALL"]` and only ignores `COM812` globally. Per-file ignores for `tests/**` disable annotation requirements, assert bans, and docstring requirements.
- `atk.providers` uses `any-llm-sdk` which re-exports OpenAI SDK types. The `openai` package is a transitive dependency — do not import it directly in `atk.providers`; always use `any_llm.types.completion` or construct dicts for message format.
