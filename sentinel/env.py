"""Load ``.env`` so nobody has to export anything, and no key lives in source.

The friction this removes: without it, every new terminal needs
``set -a; source .env; set +a`` before the SDK can find a key, and the
temptation is to paste the key into a Python file instead -- which then gets
committed.

With it, a teammate on another machine only has to drop a ``.env`` beside the
project and everything works. ``.env`` is gitignored; the key never enters the
repository, and rotating it is a one-line edit with no code change.

Deliberately dependency-free (no python-dotenv) and deliberately
non-overriding: a real environment variable always wins over the file, so CI,
Modal secrets and a shell export are never silently shadowed by a stale file.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Searched upwards from the working directory, so running from demo/ works.
SEARCH_DEPTH = 4
DEFAULT_NAME = ".env"
_loaded: set[Path] = set()


def find_dotenv(name: str = DEFAULT_NAME, start: Path | None = None) -> Path | None:
    """Nearest ``.env`` at or above ``start``."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents[:SEARCH_DEPTH]):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """A small KEY=value parser: ``#`` comments, optional ``export``, quotes."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_dotenv(
    name: str = DEFAULT_NAME,
    *,
    start: Path | None = None,
    override: bool = False,
) -> list[str]:
    """Load the nearest ``.env`` into ``os.environ``. Returns the names set.

    Values already present in the environment are left alone unless
    ``override=True``. Called at most once per file per process.
    """
    path = find_dotenv(name, start)
    if path is None or (path in _loaded and not override):
        return []
    _loaded.add(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    applied: list[str] = []
    for key, value in parse_dotenv(text).items():
        if override or not os.environ.get(key):
            os.environ[key] = value
            applied.append(key)
    return applied


def ensure_api_key(var: str = "ANTHROPIC_API_KEY") -> bool:
    """True if a key is available, loading ``.env`` first if need be."""
    if os.environ.get(var):
        return True
    load_dotenv()
    return bool(os.environ.get(var))


def api_key_hint(var: str = "ANTHROPIC_API_KEY") -> str:
    """A message that says what to do, and never prints the key itself."""
    found = find_dotenv()
    where = f"found a .env at {found}" if found else "no .env file was found"
    return (
        f"{var} is not set ({where}).\n"
        f"  Put your key in a .env file beside the project:\n"
        f"      cp .env.example .env    # then edit it\n"
        f"  or export it in your shell:\n"
        f"      export {var}=...\n"
        f"  AI generation requires this credential. A fixed template is only "
        f"available in workflows that allow template fallback; main.py does not."
    )
