"""Profile integrity analysis for suspicious candidate signals."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime
from functools import lru_cache
from typing import Callable, Iterable

from src.core.config import RankingConfig
from src.core.models import Candidate, ProfileIntegrityResult
from src.intelligence.skill_mapper import SkillTaxonomy


class ProfileIntegrityAnalyzer:
    """Detects suspicious candidate profiles with explainable penalties."""

    AI_SKILLS = {
        "llm",
        "rag",
        "embeddings",
        "retrieval",
        "ranking",
        "vector_db",
        "pytorch",
        "tensorflow",
        "huggingface",
        "transformers",
        "sentence_transformers",
        "langchain",
        "machine_learning",
        "deep_learning",
    }
    AI_EXPERIENCE_TERMS = (
        "ai",
        "machine learning",
        "ml",
        "llm",
        "rag",
        "embedding",
        "model",
        "neural",
        "deep learning",
        "pytorch",
        "tensorflow",
        "retrieval",
        "ranking",
    )
    TITLE_LEVELS = {
        "intern": 0,
        "trainee": 0,
        "junior": 1,
        "associate": 1,
        "engineer": 2,
        "developer": 2,
        "senior": 3,
        "lead": 4,
        "principal": 5,
        "staff": 5,
        "manager": 5,
        "director": 6,
        "vp": 7,
        "head": 7,
        "cto": 8,
    }
    SKILL_FAMILIES = {
        "ai_ml": AI_SKILLS,
        "backend": {"python", "fastapi", "flask", "backend"},
        "data": {"sql", "spark", "kafka", "airflow"},
        "search": {"elasticsearch", "opensearch", "faiss", "milvus", "qdrant", "weaviate", "pinecone"},
        "cloud": {"aws", "gcp", "azure", "kubernetes"},
    }

    def __init__(
        self,
        config: RankingConfig | None = None,
        taxonomy: SkillTaxonomy | None = None,
        today_provider: Callable[[], date] | None = None,
    ):
        self.config = config or RankingConfig()
        self.taxonomy = taxonomy or SkillTaxonomy()
        self._today_provider = today_provider or date.today

    def analyze(self, candidate: Candidate) -> ProfileIntegrityResult:
        penalties: list[float] = []
        reasons: list[str] = []

        self._keyword_stuffing(candidate, penalties, reasons)
        self._unrelated_skills(candidate, penalties, reasons)
        self._ai_skills_without_ai_experience(candidate, penalties, reasons)
        self._unrealistic_promotions(candidate, penalties, reasons)
        self._contradictory_career_history(candidate, penalties, reasons)
        self._suspicious_behavior(candidate, penalties, reasons)

        penalty = min(sum(penalties), 0.85)
        integrity_score = round(max(0.0, 1.0 - penalty), self.config.score_precision)
        suspicious = integrity_score < self.config.profile_integrity_suspicious_threshold
        if not reasons:
            reasons.append("No suspicious profile integrity signals detected.")
        return ProfileIntegrityResult(
            integrity_score=integrity_score,
            suspicious=suspicious,
            reasons=reasons,
        )

    def _keyword_stuffing(self, candidate: Candidate, penalties: list[float], reasons: list[str]) -> None:
        text = self._profile_text(candidate)
        term_counts = Counter()
        for alias, canonical in self.taxonomy.aliases.items():
            count = len(self._alias_pattern(alias).findall(text))
            if count:
                term_counts[canonical] += count
        repeated = sorted(term for term, count in term_counts.items() if count >= 6)
        known_mentions = sum(term_counts.values())
        unique_known = len(term_counts)
        if repeated:
            penalties.append(min(0.04 * len(repeated), 0.18))
            reasons.append("Keyword stuffing detected through repeated skill mentions: " + ", ".join(repeated[:6]) + ".")
        if known_mentions >= 50 and unique_known <= 12:
            penalties.append(0.12)
            reasons.append("Dense repeated technical keywords appear disproportionate to the profile narrative.")

    def _unrelated_skills(self, candidate: Candidate, penalties: list[float], reasons: list[str]) -> None:
        normalized_skills = {self.taxonomy.normalize(skill.name) for skill in candidate.skills}
        families = [
            name
            for name, family_skills in self.SKILL_FAMILIES.items()
            if normalized_skills & family_skills
        ]
        unknown_skills = {
            skill
            for skill in normalized_skills
            if skill not in self.taxonomy.weights and all(skill not in family for family in self.SKILL_FAMILIES.values())
        }
        if len(normalized_skills) >= 18 and len(families) >= 4:
            penalties.append(0.12)
            reasons.append("Skill list spans many unrelated areas without enough focus.")
        if len(unknown_skills) >= 8:
            penalties.append(0.08)
            reasons.append("Many skills are outside the known role taxonomy, suggesting unfocused keyword coverage.")

    def _ai_skills_without_ai_experience(self, candidate: Candidate, penalties: list[float], reasons: list[str]) -> None:
        normalized_skills = {self.taxonomy.normalize(skill.name) for skill in candidate.skills}
        ai_skill_count = len(normalized_skills & self.AI_SKILLS)
        career_text = self._career_text(candidate)
        has_ai_experience = any(term in career_text for term in self.AI_EXPERIENCE_TERMS)
        if ai_skill_count >= 3 and not has_ai_experience:
            penalties.append(0.18)
            reasons.append("AI skills are listed, but AI experience is not supported by career history.")

    def _unrealistic_promotions(self, candidate: Candidate, penalties: list[float], reasons: list[str]) -> None:
        years = candidate.profile.years_of_experience
        current_level = self._title_level(candidate.profile.current_title)
        if current_level >= 5 and years < 4:
            penalties.append(0.20)
            reasons.append("Seniority appears unusually high for the stated years of experience.")
        history = sorted(
            candidate.career_history,
            key=lambda job: self._year(job.start_date) or 0,
        )
        for previous, current in zip(history, history[1:]):
            level_jump = self._title_level(current.title) - self._title_level(previous.title)
            elapsed_months = max(self._year(current.start_date) - self._year(previous.start_date), 0) * 12
            if level_jump >= 3 and elapsed_months <= 24:
                penalties.append(0.14)
                reasons.append("Career history shows an unusually fast title jump.")
                return

    def _contradictory_career_history(self, candidate: Candidate, penalties: list[float], reasons: list[str]) -> None:
        current_jobs = [job for job in candidate.career_history if job.is_current]
        if len(current_jobs) > 1:
            penalties.append(0.12)
            reasons.append("Multiple current roles are marked active.")
        if current_jobs:
            current_job = current_jobs[0]
            if candidate.profile.current_company and current_job.company:
                if candidate.profile.current_company.lower() != current_job.company.lower():
                    penalties.append(0.10)
                    reasons.append("Profile current company contradicts the active career entry.")
            if candidate.profile.current_title and current_job.title:
                profile_level = self._title_level(candidate.profile.current_title)
                job_level = self._title_level(current_job.title)
                if abs(profile_level - job_level) >= 3:
                    penalties.append(0.08)
                    reasons.append("Profile current title does not align with the active career entry.")
        total_months = sum(max(job.duration_months, 0) for job in candidate.career_history)
        stated_months = candidate.profile.years_of_experience * 12
        if stated_months and total_months > stated_months * 1.8 and len(candidate.career_history) >= 2:
            penalties.append(0.10)
            reasons.append("Career durations exceed stated experience by a large margin.")

    def _suspicious_behavior(self, candidate: Candidate, penalties: list[float], reasons: list[str]) -> None:
        signals = candidate.signals
        if signals.profile_completeness_score >= 0.95 and not (signals.verified_email or signals.verified_phone):
            penalties.append(0.10)
            reasons.append("Profile is highly complete but lacks basic verification.")
        if signals.applications_submitted_30d >= 75 and signals.recruiter_response_rate <= 0.20:
            penalties.append(0.12)
            reasons.append("Very high application volume with low recruiter response is suspicious.")
        if signals.saved_by_recruiters_30d >= 20 and signals.profile_views_received_30d <= 2:
            penalties.append(0.08)
            reasons.append("Recruiter-save activity is inconsistent with profile view volume.")
        if signals.open_to_work and signals.avg_response_time_hours >= 120:
            penalties.append(0.08)
            reasons.append("Open-to-work status conflicts with very slow response behavior.")
        if self._days_since(signals.last_active_date) >= 180 and signals.applications_submitted_30d >= 20:
            penalties.append(0.10)
            reasons.append("Recent applications conflict with stale last-active metadata.")

    def _profile_text(self, candidate: Candidate) -> str:
        return " ".join(
            [
                candidate.profile.headline,
                candidate.profile.summary,
                candidate.profile.current_title,
                candidate.profile.current_industry,
            ]
            + [skill.name for skill in candidate.skills]
            + [job.title for job in candidate.career_history]
            + [job.industry for job in candidate.career_history]
            + [job.description for job in candidate.career_history]
        ).lower()

    @staticmethod
    def _career_text(candidate: Candidate) -> str:
        return " ".join(
            f"{job.title} {job.industry} {job.description}"
            for job in candidate.career_history
        ).lower()

    @classmethod
    def _title_level(cls, title: str) -> int:
        text = title.lower()
        return max((level for term, level in cls.TITLE_LEVELS.items() if term in text), default=2)

    @staticmethod
    def _year(value: str | None) -> int:
        if not value:
            return 0
        match = re.search(r"(19|20)\d{2}", value)
        return int(match.group(0)) if match else 0

    def _days_since(self, value: str) -> int:
        parsed = self._parse_date(value)
        if parsed is None:
            return 10_000
        return (self._today_provider() - parsed).days

    @staticmethod
    def _parse_date(value: str) -> date | None:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value[:10], fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    @lru_cache(maxsize=512)
    def _alias_pattern(alias: str) -> re.Pattern[str]:
        escaped = re.escape(SkillTaxonomy._key(alias))
        escaped = escaped.replace(r"\ ", r"[\s\-_\/]+")
        return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")
