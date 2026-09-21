from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .models import EvidenceBundle


class EvidenceCache:
    """Opt-in local JSON cache for serialized evidence bundles."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def load(self, key: str) -> EvidenceBundle | None:
        path = self._path_for(key)
        if not path.is_file():
            return None
        return EvidenceBundle.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def store(self, key: str, bundle: EvidenceBundle) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        destination = self._path_for(key)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._directory,
            prefix=f".{key}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(bundle.to_dict(), temporary, ensure_ascii=False, sort_keys=True)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)

    def _path_for(self, key: str) -> Path:
        if not key or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("Evidence cache key must be a non-empty hexadecimal digest.")
        return self._directory / f"{key}.json"
