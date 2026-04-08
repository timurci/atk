# Examples

All examples use the any-llm provider integration and require an LLM provider API key.

## Prerequisites

```bash
uv sync --group examples
```

Set the appropriate environment variable for your chosen provider, e.g.:

```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

Alternatively, pass `--api-key` as a CLI argument.

## Chat with Tools

Interactive chat loop with streaming responses and tool calling support. The assistant can call tools like `calculate`, `search_path`, and `read_file` to answer questions. Responses are rendered in real-time using Rich Live display.

```bash
cd examples/
uv run -m chat.main
```

**Usage:**
- Type a message and press Enter to send it to the assistant.
- Type `exit` or `quit` to end the session.
- Tool calls and results are displayed in styled panels.
- Text responses stream token-by-token with live updating.

**CLI options:**
- `--provider` — LLM provider identifier (default: `openai`)
- `--model` — Model name (provider default if omitted)
- `--api-key` — API key (falls back to environment variable if omitted)
- `--api-base` — Base URL for the provider API

## Structured Output

Single-call example that requests structured JSON output from the model and exits.

```bash
cd examples/
uv run -m structured
```

**Usage:**
- Enter a word when prompted.
- The model returns a `Word` object with syllables.
- The program prints the response and exits.

**CLI options:**
- `--provider` — LLM provider identifier (default: `openai`)
- `--model` — Model name (provider default if omitted)
- `--api-key` — API key (falls back to environment variable if omitted)
- `--api-base` — Base URL for the provider API
