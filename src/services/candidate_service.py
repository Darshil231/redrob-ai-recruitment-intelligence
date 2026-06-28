"""Candidate repository-style access for files and in-memory API payloads."""

from pathlib import Path
from typing import Iterable

from src.core.models import Candidate
from src.parsers.candidate_parser import CandidateParser


class CandidateService:
    """Loads and searches candidates without coupling callers to parser details."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self._candidates: list[Candidate] | None = None

    def list_candidates(self, refresh: bool = False) -> list[Candidate]:
        if self._candidates is None or refresh:
            self._candidates = list(CandidateParser(str(self.filepath)).parse())
        return self._candidates

    def get_candidate(self, candidate_id: str) -> Candidate | None:
        return next((candidate for candidate in self.list_candidates() if candidate.candidate_id == candidate_id), None)

    def search(self, query: str, limit: int = 25) -> list[Candidate]:
        query_lower = query.lower()
        matches = []
        for candidate in self.list_candidates():
            haystack = " ".join(
                [
                    candidate.profile.anonymized_name,
                    candidate.profile.current_title,
                    candidate.profile.current_company,
                    candidate.profile.summary,
                    " ".join(skill.name for skill in candidate.skills),
                ]
            ).lower()
            if query_lower in haystack:
                matches.append(candidate)
                if len(matches) >= limit:
                    break
        return matches

    @staticmethod
    def from_iterable(candidates: Iterable[Candidate]) -> list[Candidate]:
        return list(candidates)
