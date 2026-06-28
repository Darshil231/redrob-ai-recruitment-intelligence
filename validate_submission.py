"""Validate generated recruitment submission CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]
EXPECTED_ROWS = 100


class SubmissionValidationError(ValueError):
    """Raised when a submission file violates the expected contract."""


def validate_submission(path: str | Path) -> bool:
    submission_path = Path(path)
    if not submission_path.exists():
        raise SubmissionValidationError(f"Submission file does not exist: {submission_path}")

    with submission_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise SubmissionValidationError(
                f"Invalid columns {reader.fieldnames}; expected {REQUIRED_COLUMNS}."
            )
        rows = list(reader)

    if len(rows) != EXPECTED_ROWS:
        raise SubmissionValidationError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}.")

    seen_candidate_ids: set[str] = set()
    parsed: list[tuple[float, str, int]] = []
    for index, row in enumerate(rows, start=1):
        candidate_id = row["candidate_id"].strip()
        reasoning = row["reasoning"].strip()
        if not candidate_id:
            raise SubmissionValidationError(f"Row {index} has an empty candidate_id.")
        if candidate_id in seen_candidate_ids:
            raise SubmissionValidationError(f"Duplicate candidate_id: {candidate_id}.")
        seen_candidate_ids.add(candidate_id)
        if not reasoning:
            raise SubmissionValidationError(f"Row {index} has empty reasoning.")

        try:
            rank = int(row["rank"])
        except ValueError as exc:
            raise SubmissionValidationError(f"Row {index} has non-integer rank: {row['rank']}.") from exc
        if rank != index:
            raise SubmissionValidationError(f"Row {index} rank is {rank}; expected {index}.")

        try:
            score = float(row["score"])
        except ValueError as exc:
            raise SubmissionValidationError(f"Row {index} has non-numeric score: {row['score']}.") from exc
        if not 0.0 <= score <= 1.0:
            raise SubmissionValidationError(f"Row {index} score {score} is outside [0, 1].")

        parsed.append((score, candidate_id, rank))

    expected_order = sorted(parsed, key=lambda item: (-item[0], item[1]))
    if parsed != expected_order:
        raise SubmissionValidationError(
            "Rows must be sorted by score descending and candidate_id ascending for ties."
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission.csv format and ordering.")
    parser.add_argument("path", nargs="?", default="submission.csv")
    args = parser.parse_args()
    validate_submission(args.path)
    print(f"Validated {args.path}: {EXPECTED_ROWS} rows.")


if __name__ == "__main__":
    main()
