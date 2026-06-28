"""Role relevance scoring against the target job description."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from src.core.models import Candidate, JobDNA, RoleRelevanceResult


class RoleRelevanceAnalyzer:
    """Scores whether a candidate's career track fits the target role."""

    STRONG_ROLE_TERMS = (
        "machine learning",
        "ml engineer",
        "ai engineer",
        "data scientist",
        "data engineer",
        "applied scientist",
        "nlp engineer",
        "research scientist",
        "computer vision",
        "mlops",
        "ai platform",
    )
    ADJACENT_ROLE_TERMS = (
        "software engineer",
        "backend engineer",
        "full stack",
        "platform engineer",
        "devops",
        "analytics engineer",
        "business analyst",
        "data analyst",
    )
    UNRELATED_ROLE_TERMS = (
        "civil engineer",
        "mechanical engineer",
        "marketing manager",
        "hr",
        "human resources",
        "sales",
        "customer support",
        "content writer",
    )
    AI_TRANSITION_TERMS = (
        "machine learning",
        "deep learning",
        "llm",
        "rag",
        "nlp",
        "computer vision",
        "embeddings",
        "vector database",
        "retrieval",
        "ranking",
        "model serving",
        "production ml",
        "fine-tuning",
        "pytorch",
        "tensorflow",
        "langchain",
        "hugging face",
        "ml pipeline",
    )
    DATA_INDUSTRY_TERMS = (
        "ai",
        "artificial intelligence",
        "machine learning",
        "data",
        "analytics",
        "software",
        "saas",
        "cloud",
        "it services",
        "hr tech",
        "recruitment",
    )

    def analyze(self, candidate: Candidate, job_dna: JobDNA) -> RoleRelevanceResult:
        current_title = candidate.profile.current_title.lower()
        previous_titles = [job.title.lower() for job in candidate.career_history]
        industries = self._industries(candidate)
        career_text = self._career_text(candidate)

        current_title_score = self._title_score(current_title, job_dna)
        previous_title_score = max((self._title_score(title, job_dna) for title in previous_titles), default=0.0)
        industry_score = self._industry_score(industries)
        progression_score = self._progression_score(current_title, previous_titles)
        transition_score = self._transition_score(candidate, career_text)
        unrelated_penalty = self._unrelated_penalty(current_title, previous_titles, transition_score)

        raw_score = (
            current_title_score * 0.38
            + previous_title_score * 0.18
            + industry_score * 0.16
            + progression_score * 0.12
            + transition_score * 0.16
        )
        score = self._clamp(raw_score - unrelated_penalty)
        return RoleRelevanceResult(
            score=round(score, 3),
            current_title_score=round(current_title_score, 3),
            previous_title_score=round(previous_title_score, 3),
            industry_score=round(industry_score, 3),
            progression_score=round(progression_score, 3),
            transition_score=round(transition_score, 3),
            unrelated_penalty=round(unrelated_penalty, 3),
            reasoning=self._reasoning(
                candidate,
                score,
                current_title_score,
                previous_title_score,
                industry_score,
                progression_score,
                transition_score,
                unrelated_penalty,
            ),
        )

    def _title_score(self, title: str, job_dna: JobDNA) -> float:
        if not title:
            return 0.0
        if self._contains_any(title, self.STRONG_ROLE_TERMS):
            return 1.0
        if self._contains_any(title, self.UNRELATED_ROLE_TERMS):
            return 0.05
        if self._contains_any(title, job_dna.title_keywords):
            return 0.90
        if self._contains_any(title, self.ADJACENT_ROLE_TERMS):
            return 0.60
        return 0.30

    def _industry_score(self, industries: Iterable[str]) -> float:
        text = " ".join(industries).lower()
        if self._contains_any(text, ("ai", "machine learning", "data", "analytics")):
            return 1.0
        if self._contains_any(text, self.DATA_INDUSTRY_TERMS):
            return 0.70
        if text:
            return 0.25
        return 0.0

    def _progression_score(self, current_title: str, previous_titles: list[str]) -> float:
        titles = previous_titles + [current_title]
        strong_positions = [index for index, title in enumerate(titles) if self._contains_any(title, self.STRONG_ROLE_TERMS)]
        adjacent_positions = [index for index, title in enumerate(titles) if self._contains_any(title, self.ADJACENT_ROLE_TERMS)]
        unrelated_current = self._contains_any(current_title, self.UNRELATED_ROLE_TERMS)
        if strong_positions and strong_positions[-1] == len(titles) - 1:
            return 1.0
        if strong_positions:
            return 0.75
        if adjacent_positions and not unrelated_current:
            return 0.55
        return 0.20

    def _transition_score(self, candidate: Candidate, career_text: str) -> float:
        skills_text = " ".join(skill.name for skill in candidate.skills).lower()
        summary_text = f"{candidate.profile.headline} {candidate.profile.summary}".lower()
        evidence_hits = 0
        for text in (skills_text, summary_text, career_text):
            hits = sum(1 for term in self.AI_TRANSITION_TERMS if term in text)
            if hits:
                evidence_hits += min(hits, 4)
        return self._clamp(evidence_hits / 8)

    def _unrelated_penalty(self, current_title: str, previous_titles: list[str], transition_score: float) -> float:
        current_unrelated = self._contains_any(current_title, self.UNRELATED_ROLE_TERMS)
        history_unrelated = any(self._contains_any(title, self.UNRELATED_ROLE_TERMS) for title in previous_titles)
        if current_unrelated and transition_score < 0.35:
            return 0.45
        if current_unrelated:
            return 0.20
        if history_unrelated and transition_score < 0.25:
            return 0.15
        return 0.0

    def _reasoning(
        self,
        candidate: Candidate,
        score: float,
        current_title_score: float,
        previous_title_score: float,
        industry_score: float,
        progression_score: float,
        transition_score: float,
        unrelated_penalty: float,
    ) -> list[str]:
        reasons = [f"Role relevance score is {score:.3f}."]
        if current_title_score >= 0.85:
            reasons.append(f"Current title '{candidate.profile.current_title}' strongly matches the target role family.")
        elif current_title_score <= 0.10:
            reasons.append(f"Current title '{candidate.profile.current_title}' is outside the AI/ML/Data role family.")
        if previous_title_score >= 0.75:
            reasons.append("Previous title history includes AI/ML/Data role alignment.")
        if industry_score >= 0.70:
            reasons.append("Industry history is relevant to AI, data, software, or recruiting technology.")
        if progression_score >= 0.75:
            reasons.append("Career progression moves toward the target role.")
        if transition_score >= 0.35:
            reasons.append("Profile contains evidence of transition into AI/ML through skills, summary, or project work.")
        if unrelated_penalty:
            reasons.append("Unrelated profession penalty applied because role history is not clearly AI/ML/Data.")
        return reasons

    @staticmethod
    def _industries(candidate: Candidate) -> list[str]:
        values = [candidate.profile.current_industry]
        values.extend(job.industry for job in candidate.career_history)
        return [value for value in values if value]

    @staticmethod
    def _career_text(candidate: Candidate) -> str:
        parts = [
            candidate.profile.headline,
            candidate.profile.summary,
            candidate.profile.current_title,
            candidate.profile.current_industry,
        ]
        parts.extend(f"{job.title} {job.industry} {job.description}" for job in candidate.career_history)
        return " ".join(parts).lower()

    @staticmethod
    def _contains_any(text: str, terms: Iterable[str]) -> bool:
        return any(term and RoleRelevanceAnalyzer._term_pattern(term).search(text) for term in terms)

    @staticmethod
    @lru_cache(maxsize=512)
    def _term_pattern(term: str) -> re.Pattern[str]:
        return re.compile(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])")

    @staticmethod
    def _clamp(score: float) -> float:
        return min(max(score, 0.0), 1.0)
