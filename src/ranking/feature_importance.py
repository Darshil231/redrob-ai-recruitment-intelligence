"""Feature contribution analysis for candidate ranking explanations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.core.config import RankingConfig
from src.core.models import ProfileIntegrityResult
from src.ranking.rankers import EnsembleRankingResult


@dataclass(frozen=True)
class FeatureContribution:
    """Single feature contribution with a recruiter-readable visualization."""

    feature: str
    contribution: float
    percentage: float
    visualization: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureImportanceResult:
    """Feature-level attribution for a ranked candidate."""

    features: list[FeatureContribution]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": [feature.to_dict() for feature in self.features],
            "explanation": self.explanation,
        }


class FeatureImportance:
    """Calculates contribution percentages for ranking components."""

    COMPONENT_LABELS = {
        "skill": "Skill",
        "career": "Career",
        "behavior": "Behavior",
        "role_relevance": "Role Relevance",
        "semantic": "Semantic",
        "eligibility": "Eligibility",
        "integrity": "Integrity",
    }
    COMPONENT_WEIGHTS = {
        "skill": "skill_weight",
        "career": "career_weight",
        "behavior": "behavior_weight",
        "role_relevance": "role_relevance_weight",
        "semantic": "semantic_weight",
        "eligibility": "eligibility_weight",
    }

    def __init__(self, config: RankingConfig | None = None):
        self.config = config or RankingConfig()

    def analyze(
        self,
        ensemble: EnsembleRankingResult,
        profile_integrity: ProfileIntegrityResult | Mapping[str, Any],
    ) -> FeatureImportanceResult:
        integrity_score = self._integrity_score(profile_integrity)
        raw = self._component_contributions(ensemble)
        raw["integrity"] = -(
            (1.0 - self._clamp(integrity_score))
            * self.config.profile_integrity_penalty_weight
        )
        denominator = sum(abs(value) for value in raw.values())
        features = [
            self._feature(name, value, denominator)
            for name, value in raw.items()
        ]
        features.sort(key=lambda item: abs(item.contribution), reverse=True)
        return FeatureImportanceResult(
            features=features,
            explanation=self._explanation(features),
        )

    def _component_contributions(self, ensemble: EnsembleRankingResult) -> dict[str, float]:
        contributions = {}
        for component, field_name in self.COMPONENT_WEIGHTS.items():
            result = ensemble.component_results.get(component)
            if result is None:
                contributions[component] = 0.0
                continue
            weight = max(getattr(self.config, field_name), self.config.score_floor)
            if self.config.ensemble_confidence_weighting:
                weight *= self._clamp(result.confidence)
            contributions[component] = self._clamp(result.score) * weight
        return contributions

    def _feature(self, name: str, contribution: float, denominator: float) -> FeatureContribution:
        percentage = 0.0 if denominator <= 0 else (contribution / denominator) * 100
        label = self.COMPONENT_LABELS[name]
        return FeatureContribution(
            feature=label,
            contribution=round(contribution, self.config.score_precision),
            percentage=round(percentage, 1),
            visualization=self._bar(label, percentage),
        )

    def _bar(self, label: str, percentage: float) -> str:
        width = 20
        blocks = round(min(abs(percentage), 100.0) / 100 * width)
        bar = "#" * blocks + "." * (width - blocks)
        sign = "-" if percentage < 0 else "+"
        return f"{label:<11} | {bar} {sign}{abs(percentage):.1f}%"

    @staticmethod
    def _explanation(features: list[FeatureContribution]) -> str:
        visible = [feature for feature in features if feature.percentage != 0]
        if not visible:
            return "No feature contribution was available."
        return "Feature contribution mix: " + ", ".join(
            f"{feature.feature} {feature.percentage:+.1f}%"
            for feature in visible
        ) + "."

    @staticmethod
    def _integrity_score(profile_integrity: ProfileIntegrityResult | Mapping[str, Any]) -> float:
        if isinstance(profile_integrity, ProfileIntegrityResult):
            return profile_integrity.integrity_score
        return float(profile_integrity.get("integrity_score", 1.0))

    def _clamp(self, score: float) -> float:
        return min(max(score, self.config.score_floor), self.config.score_ceiling)
