"""Candidate parsing service for JSONL datasets and API-provided dictionaries."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from src.core.models import (
    Candidate,
    Profile,
    CareerEntry,
    Education,
    Skill,
    Language,
    Certification,
    Signals,
    SalaryRange,
)

try:
    import orjson
except ImportError:  # pragma: no cover - optional speed path
    orjson = None


class CandidateParser:

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def parse(self) -> Iterator[Candidate]:
        loads = orjson.loads if orjson is not None else json.loads
        build_candidate = self._build_candidate
        if self.filepath.suffix.lower() == ".json":
            raw = loads(self.filepath.read_bytes())
            records = raw if isinstance(raw, list) else raw.get("candidates", [])
            for record in records:
                yield build_candidate(record)
            return

        with self.filepath.open("rb", buffering=1024 * 1024) as f:
            for line in f:
                if not line.strip():
                    continue

                yield build_candidate(loads(line))

    def parse_many(self, records: Iterable[Dict[str, Any]]) -> List[Candidate]:
        return [self._build_candidate(record) for record in records]

    @classmethod
    def from_records(cls, records: Iterable[Dict[str, Any]]) -> List[Candidate]:
        parser = cls(filepath="")
        return parser.parse_many(records)

    def _build_candidate(self, raw: Dict[str, Any]) -> Candidate:

        # ---------- Profile ----------
        p = raw.get("profile", {})

        profile = Profile(
            headline=p.get("headline", ""),
            summary=p.get("summary", ""),
            current_title=p.get("current_title", ""),
            current_company=p.get("current_company", ""),
            current_industry=p.get("current_industry", ""),
            current_company_size=p.get("current_company_size", ""),
            years_of_experience=p.get("years_of_experience", 0),
            location=p.get("location", ""),
            country=p.get("country", ""),
            anonymized_name=p.get("anonymized_name", ""),
        )

        # ---------- Career ----------
        career = []

        for c in raw.get("career_history", []):
            career.append(
                CareerEntry(
                    company=c.get("company", ""),
                    title=c.get("title", ""),
                    start_date=c.get("start_date", ""),
                    end_date=c.get("end_date"),
                    duration_months=c.get("duration_months", 0),
                    is_current=c.get("is_current", False),
                    industry=c.get("industry", ""),
                    company_size=c.get("company_size", ""),
                    description=c.get("description", ""),
                )
            )

        # ---------- Education ----------
        education = []

        for e in raw.get("education", []):
            education.append(
                Education(
                    institution=e.get("institution", ""),
                    degree=e.get("degree", ""),
                    field_of_study=e.get("field_of_study", ""),
                    start_year=e.get("start_year", 0),
                    end_year=e.get("end_year", 0),
                    grade=e.get("grade", ""),
                    tier=e.get("tier", ""),
                )
            )

        # ---------- Skills ----------
        skills = []

        for s in raw.get("skills", []):
            skills.append(
                Skill(
                    name=s.get("name", ""),
                    proficiency=s.get("proficiency", ""),
                    endorsements=s.get("endorsements", 0),
                    duration_months=s.get("duration_months", 0),
                )
            )

        # ---------- Languages ----------
        languages = []

        for l in raw.get("languages", []):
            languages.append(
                Language(
                    language=l.get("language", ""),
                    proficiency=l.get("proficiency", ""),
                )
            )

        # ---------- Certifications ----------
        certifications = []

        for cert in raw.get("certifications", []):
            certifications.append(
                Certification(
                    name=cert.get("name", ""),
                    issuer=cert.get("issuer", ""),
                    year=cert.get("year"),
                )
            )

        # ---------- Signals ----------
        rs = raw.get("redrob_signals") or raw.get("signals", {})

        salary = SalaryRange(
            minimum=self._salary_value(rs, "min", "minimum"),
            maximum=self._salary_value(rs, "max", "maximum"),
        )

        signals = Signals(
            open_to_work=rs.get("open_to_work_flag", rs.get("open_to_work", False)),
            notice_period_days=rs.get("notice_period_days", 90),
            recruiter_response_rate=rs.get("recruiter_response_rate", 0),
            interview_completion_rate=rs.get("interview_completion_rate", 0),
            github_activity_score=rs.get("github_activity_score", 0),
            profile_completeness_score=rs.get(
                "profile_completeness_score", 0
            ),
            offer_acceptance_rate=rs.get("offer_acceptance_rate", 0),
            avg_response_time_hours=rs.get("avg_response_time_hours", 0),
            applications_submitted_30d=rs.get(
                "applications_submitted_30d", 0
            ),
            profile_views_received_30d=rs.get(
                "profile_views_received_30d", 0
            ),
            search_appearance_30d=rs.get(
                "search_appearance_30d", 0
            ),
            saved_by_recruiters_30d=rs.get(
                "saved_by_recruiters_30d", 0
            ),
            preferred_work_mode=rs.get("preferred_work_mode", ""),
            willing_to_relocate=rs.get("willing_to_relocate", False),
            verified_email=rs.get("verified_email", False),
            verified_phone=rs.get("verified_phone", False),
            connection_count=rs.get("connection_count", 0),
            endorsements_received=rs.get(
                "endorsements_received", 0
            ),
            linkedin_connected=rs.get("linkedin_connected", False),
            last_active_date=rs.get("last_active_date", ""),
            signup_date=rs.get("signup_date", ""),
            expected_salary=salary,
            skill_assessment_scores=rs.get(
                "skill_assessment_scores", {}
            ),
        )

        return Candidate(
            candidate_id=raw.get("candidate_id", ""),
            profile=profile,
            career_history=career,
            education=education,
            skills=skills,
            certifications=certifications,
            languages=languages,
            signals=signals,
        )

    @staticmethod
    def _salary_value(raw_signals: Dict[str, Any], raw_key: str, model_key: str) -> float:
        salary = raw_signals.get("expected_salary_range_inr_lpa") or raw_signals.get("expected_salary") or {}
        return salary.get(raw_key, salary.get(model_key, 0))
