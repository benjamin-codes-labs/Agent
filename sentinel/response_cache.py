from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any


CACHE_VERSION = 1


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def response_cache_key(request: Any, *, code_version: str | None = None) -> str:
    payload = {"version": CACHE_VERSION, "code": code_version or implementation_fingerprint(), "request": request}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, directory: str | Path, *, max_age_seconds: float = 86400):
        if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("cache age must be positive and finite")
        self.directory = Path(directory)
        self.max_age_seconds = max_age_seconds

    def _path(self, key: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("cache key must be a SHA256 hex digest")
        return self.directory / f"{key}.json"

    def get(self, key: str, *, now: float | None = None) -> dict | None:
        path = self._path(key)
        try:
            if path.stat().st_size > 8_000_000:
                return None
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict) or envelope.get("version") != CACHE_VERSION or envelope.get("key") != key:
                return None
            created = envelope.get("created_at")
            if isinstance(created, bool) or not isinstance(created, (int, float)) or not math.isfinite(created):
                return None
            age = (time.time() if now is None else now) - created
            if age < 0 or age > self.max_age_seconds:
                return None
            data = envelope.get("data")
            if not isinstance(data, dict):
                return None
            if hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest() != envelope.get("checksum"):
                return None
            return data
        except (OSError, ValueError, TypeError, RecursionError):
            return None

    def put(self, key: str, data: dict, *, now: float | None = None) -> None:
        path = self._path(key)
        body = _canonical(data)
        envelope = {"version": CACHE_VERSION, "key": key, "created_at": time.time() if now is None else now,
                    "checksum": hashlib.sha256(body.encode("utf-8")).hexdigest(), "data": data}
        encoded = _canonical(envelope)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             prefix=".response-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
