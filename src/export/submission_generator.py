"""Competition-style submission CSV generation."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from src.core.models import RankedCandidate
from validate_submission import EXPECTED_ROWS, REQUIRED_COLUMNS, SubmissionValidationError


class SubmissionGenerator:
    """Writes and validates the final 100-row ranked candidate submission."""

    columns = REQUIRED_COLUMNS

    def __init__(self, row_count: int = EXPECTED_ROWS):
        self.row_count = row_count

    def generate(self, results: Iterable[RankedCandidate], path: str | Path = "submission.csv") -> Path:
        ranked = self._sorted_results(results)
        if len(ranked) < self.row_count:
            raise ValueError(f"Need at least {self.row_count} ranked candidates, found {len(ranked)}.")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            for rank, result in enumerate(ranked[: self.row_count], start=1):
                writer.writerow(self._row(result, rank))

        self._run_validator(output_path)
        return output_path

    @staticmethod
    def _sorted_results(results: Iterable[RankedCandidate]) -> list[RankedCandidate]:
        return sorted(
            results,
            key=lambda result: (-SubmissionGenerator._submission_score(result), result.candidate.candidate_id),
        )

    @staticmethod
    def _row(result: RankedCandidate, rank: int) -> dict[str, object]:
        return {
            "candidate_id": result.candidate.candidate_id,
            "rank": rank,
            "score": f"{result.final_score:.6f}",
            "reasoning": SubmissionGenerator._reasoning(result),
        }

    @staticmethod
    def _submission_score(result: RankedCandidate) -> float:
        return float(f"{result.final_score:.6f}")

    @staticmethod
    def _run_validator(output_path: Path) -> None:
        completed = subprocess.run(
            [sys.executable, "validate_submission.py", str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout).strip()
            raise SubmissionValidationError(message or f"Validation failed for {output_path}.")

    @staticmethod
    def _reasoning(result: RankedCandidate) -> str:
        parts = [result.recommendation]
        parts.extend(result.why_selected[:3])
        if result.why_rejected:
            parts.append("Risks: " + " ".join(result.why_rejected[:2]))
        return " ".join(part.strip() for part in parts if part and part.strip())
