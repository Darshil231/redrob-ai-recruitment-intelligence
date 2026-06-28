"""Career-signal analysis for recruiter-grade candidate assessment."""

from typing import Dict, Iterable, List

from src.core.models import Candidate, CareerAnalysisResult


class CareerAnalyzer:
    """Detects career patterns that correlate with fit for AI engineering roles."""

    SIGNAL_TERMS: Dict[str, tuple[str, ...]] = {
        "production_ml": ("production", "deployed", "serving", "model", "ml pipeline", "feature pipeline", "llm"),
        "backend": ("backend", "api", "fastapi", "flask", "microservice", "distributed system"),
        "big_data": ("spark", "kafka", "beam", "airflow", "snowflake", "data lake"),
        "leadership": ("led", "lead", "mentored", "owned", "architected", "staff"),
        "management": ("managed", "hired", "performance review", "team of"),
        "ai_research": ("research", "paper", "publication", "novel", "experiment"),
        "mlops": ("mlops", "monitoring", "model registry", "ci/cd", "kubeflow"),
        "cloud": ("aws", "gcp", "azure", "kubernetes", "docker"),
        "startup": ("startup", "founding", "0 to 1", "early stage"),
    }

    WEIGHTS: Dict[str, float] = {
        "production_ml": 0.22,
        "backend": 0.16,
        "big_data": 0.14,
        "leadership": 0.12,
        "management": 0.08,
        "ai_research": 0.08,
        "mlops": 0.08,
        "cloud": 0.07,
        "startup": 0.05,
    }

    def analyze(self, candidate: Candidate) -> CareerAnalysisResult:
        signals = {name: 0.0 for name in self.SIGNAL_TERMS}
        relevant_experience: List[str] = []
        relevant_projects: List[str] = []

        for job in candidate.career_history:
            text = self._job_text(job.title, job.description, job.industry, job.company_size)
            for signal, terms in self.SIGNAL_TERMS.items():
                if self._contains_any(text, terms):
                    signals[signal] = min(signals[signal] + self._job_contribution(job.duration_months), 1.0)
                    relevant_experience.append(f"{job.title} at {job.company}")
            if self._contains_any(text, ("built", "launched", "deployed", "migrated", "scaled")):
                relevant_projects.append(job.description[:220])

        score = sum(signals[name] * self.WEIGHTS[name] for name in self.WEIGHTS)
        experience_bonus = min(candidate.profile.years_of_experience / 100, 0.08)
        score = min(score + experience_bonus, 1.0)
        reasoning = self._reasoning(signals, candidate.profile.years_of_experience)

        return CareerAnalysisResult(
            score=round(score, 3),
            signals={key: round(value, 3) for key, value in signals.items()},
            relevant_experience=sorted(set(relevant_experience))[:8],
            relevant_projects=[project for project in relevant_projects if project][:5],
            reasoning=reasoning,
        )

    @staticmethod
    def _job_text(*parts: str) -> str:
        return " ".join(part or "" for part in parts).lower()

    @staticmethod
    def _contains_any(text: str, terms: Iterable[str]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _job_contribution(duration_months: int) -> float:
        return max(0.20, min(duration_months / 36, 1.0))

    @staticmethod
    def _reasoning(signals: Dict[str, float], years: float) -> List[str]:
        strongest = [name.replace("_", " ") for name, score in signals.items() if score >= 0.5]
        reasoning = []
        if strongest:
            reasoning.append("Strong career signals: " + ", ".join(strongest[:5]) + ".")
        if years:
            reasoning.append(f"Profile reports {years:.1f} years of experience.")
        return reasoning
