"""CSV export for ranked candidate shortlists."""

import csv
from pathlib import Path
from typing import Iterable

from src.core.models import RankedCandidate


class CSVExporter:
    """Writes recruiter-readable ranking summaries."""

    columns = [
        "Candidate ID",
        "Name",
        "Company",
        "Title",
        "Experience",
        "Skill Score",
        "Career Score",
        "Behavior Score",
        "Semantic Score",
        "Final Score",
        "Recommendation",
        "Explanation",
    ]

    def export(self, results: Iterable[RankedCandidate], path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            for result in results:
                writer.writerow(self._row(result))
        return output_path

    def _row(self, result: RankedCandidate) -> dict[str, object]:
        candidate = result.candidate
        return {
            "Candidate ID": candidate.candidate_id,
            "Name": candidate.profile.anonymized_name,
            "Company": candidate.profile.current_company,
            "Title": candidate.profile.current_title,
            "Experience": candidate.profile.years_of_experience,
            "Skill Score": result.skill_score,
            "Career Score": result.career_score,
            "Behavior Score": result.behavior_score,
            "Semantic Score": result.semantic_score,
            "Final Score": result.final_score,
            "Recommendation": result.recommendation,
            "Explanation": " ".join(result.why_selected[:4]),
        }
