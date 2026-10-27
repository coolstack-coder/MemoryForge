"""Pull a function definition out of free-form model output."""

from __future__ import annotations

import ast
import re

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# A tripwire, not a security boundary: isolation comes from the subprocess,
# rlimits and (in Docker) the container.
_BANNED_CALLS = {"system", "popen", "fork", "execv", "execve", "spawn", "rmtree", "remove", "unlink"}
_BANNED_MODULES = {"subprocess", "socket", "shutil", "ctypes", "multiprocessing", "requests", "urllib", "http"}


class ExtractionError(ValueError):
    pass


def extract_function(text: str, entry_point: str) -> str:
    blocks = _FENCE_RE.findall(text) or [text]
    # prefer the last block that defines the entry point
    for block in reversed(blocks):
        src = block.strip()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        if any(isinstance(n, ast.FunctionDef) and n.name == entry_point for n in tree.body):
            _guard(tree)
            return src
    raise ExtractionError(f"no parseable definition of `{entry_point}` found")


def _guard(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".")[0] in _BANNED_MODULES:
                    raise ExtractionError(f"import of `{name}` is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in _BANNED_CALLS:
            raise ExtractionError(f"call to `.{node.attr}` is not allowed")
