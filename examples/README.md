# Examples

All examples use the OpenAI language model and require an OpenAI-compatible API endpoint.

## Prerequisites

```bash
uv sync --group examples
```

## Chat with Tools

Interactive chat loop with tool calling support. The assistant can call tools like `calculate`, `search_path`, and `read_file` to answer questions.

```bash
cd examples/
uv run -m chat.main
```

**Usage:**
- Type a message and press Enter to send it to the assistant.
- Type `exit` or `quit` to end the session.
- Tool calls and results are displayed in styled panels.

**CLI options:**
- `--base-url` — API base URL (default: `http://localhost:8080/v1`)
- `--api-key` — API key (default: empty)
- `--model` — Model name (default: `gpt-4o-mini`, ignored in local setup)

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
- `--base-url` — API base URL (default: `http://localhost:8080/v1`)
- `--api-key` — API key (default: empty)
- `--model` — Model name (default: `gpt-4o-mini`, ignored in local setup)
