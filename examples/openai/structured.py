"""OpenAI-compatible structured output chat example."""

from pydantic import BaseModel

from atk.openai.model import OpenAILanguageModel
from examples.openai.chat import chat_loop, parse_args

instruction = "You are a helpful assistant. You help users write syllables of a word."


class Word(BaseModel):
    """A word with syllables."""

    syllables: list[str]


def main() -> None:
    """Run chat loop with structured output."""
    args = parse_args()
    llm = OpenAILanguageModel(
        base_url=args["base_url"],
        api_key=args["api_key"],
        model=args["model"],
    )
    print(f'System: "{instruction}"\n')  # noqa: T201
    chat_loop(llm, instruction, output_schema=Word)


if __name__ == "__main__":
    main()
