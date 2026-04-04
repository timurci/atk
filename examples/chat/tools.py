"""Example tools for the chat demo."""

import fnmatch
import os
from pathlib import Path
from typing import Literal


def calculate(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: float,
    b: float,
) -> str:
    """Perform a mathematical calculation.

    Args:
        operation: The operation to perform.
        a: First number.
        b: Second number.
    """
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y,
    }
    result = ops[operation](a, b)
    return f"{a} {operation} {b} = {result}"


def search_path(pattern: str, directory: str = ".", max_depth: int = 1) -> str:
    """Search for files and directories matching a glob pattern.

    Args:
        pattern: Glob pattern to match, e.g. ``*.py`` or ``test_*``.
        directory: Directory to search in. Defaults to current directory.
        max_depth: Maximum directory depth to search. Defaults to 1 (no recursion).
    """
    path = Path(directory).expanduser()
    entries: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(path):
        depth = len(Path(dirpath).relative_to(path).parts)
        if depth >= max_depth:
            dirnames.clear()
            continue
        entries.extend(
            ("d", str(Path(dirpath) / dirname))
            for dirname in dirnames
            if fnmatch.fnmatch(dirname, pattern)
        )
        entries.extend(
            ("f", str(Path(dirpath) / filename))
            for filename in filenames
            if fnmatch.fnmatch(filename, pattern)
        )
    entries.sort(key=lambda e: e[1])
    if not entries:
        return f"No matches for {pattern!r} in {directory!r}"
    lines: list[str] = [f"Found {len(entries)} match(es):"]
    lines.extend(f"  {prefix} {name}" for prefix, name in entries)
    return "\n".join(lines)


def read_file(path: str) -> str:
    """Read and return the contents of a file.

    Args:
        path: Path to the file to read.
    """
    return Path(path).expanduser().read_text(encoding="utf-8")
