"""JSON export for APIs and downstream analytics."""

import json
from pathlib import Path
from typing import Iterable

from src.core.models import RankedCandidate


class JSONExporter:
    """Serializes ranked candidates with full explainability payloads."""

    def export(self, results: Iterable[RankedCandidate], path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.to_dict() for result in results]
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path
