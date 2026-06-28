"""Ranking formulas and score composition for the recruitment pipeline."""

from __future__ import annotations

from collections.abc import Mapping

from src.core.config import RankingConfig
from src.core.models import CareerAnalysisResult
from src.ranking.rankers import EnsembleRanker, EnsembleRankingResult, RankerResult


class RankingEngine:
    """Single owner for configurable ranking formulas."""

    def __init__(self, config: RankingConfig | None = None):
        self.config = config or RankingConfig()
        self.ensemble_ranker = EnsembleRanker(self.config)

    def fast_score(
        self,
        skill_score: float,
        career_score: float,
        filter_score: float,
        behavior_score: float | None = None,
        role_relevance_score: float | None = None,
    ) -> float:
        components = {
            "skill": (skill_score, self.config.skill_weight),
            "career": (career_score, self.config.career_weight),
            "eligibility": (filter_score, self.config.eligibility_weight),
            "behavior": (behavior_score, self.config.behavior_weight),
            "role_relevance": (role_relevance_score, self.config.role_relevance_weight),
        }
        return self._finalize(self._weighted_average(components))

    def final_score(
        self,
        fast_score: float,
        semantic_score: float,
        reasoning_score: float | None = None,
        *,
        llm_confidence: float | None = None,
        honeypot_penalty: float | None = None,
        disqualifier_penalty: float | None = None,
        profile_integrity_penalty: float | None = None,
    ) -> float:
        reasoning_value = reasoning_score if reasoning_score is not None else llm_confidence
        components = {
            "fast": (fast_score, self.config.fast_score_weight),
            "semantic": (semantic_score, self.config.semantic_weight),
            "reasoning": (reasoning_value, self.config.reasoning_weight),
        }
        score = self._weighted_average(components)
        score = self.apply_penalties(
            score,
            honeypot_penalty=honeypot_penalty,
            disqualifier_penalty=disqualifier_penalty,
            profile_integrity_penalty=profile_integrity_penalty,
        )
        return self._finalize(score)

    def score_components(
        self,
        *,
        skill_score: float,
        career_score: float,
        behavior_score: float,
        semantic_score: float,
        eligibility_score: float,
        reasoning_score: float,
        role_relevance_score: float | None = None,
        honeypot_penalty: float | None = None,
        disqualifier_penalty: float | None = None,
        profile_integrity_penalty: float | None = None,
    ) -> float:
        score = self._weighted_average(
            {
                "skill": (skill_score, self.config.skill_weight),
                "career": (career_score, self.config.career_weight),
                "behavior": (behavior_score, self.config.behavior_weight),
                "role_relevance": (role_relevance_score, self.config.role_relevance_weight),
                "semantic": (semantic_score, self.config.semantic_weight),
                "eligibility": (eligibility_score, self.config.eligibility_weight),
                "reasoning": (reasoning_score, self.config.reasoning_weight),
            }
        )
        score = self.apply_penalties(
            score,
            honeypot_penalty=honeypot_penalty,
            disqualifier_penalty=disqualifier_penalty,
            profile_integrity_penalty=profile_integrity_penalty,
        )
        return self._finalize(score)

    def ensemble_score(
        self,
        *,
        skill_score: float | RankerResult | dict,
        career_score: float | RankerResult | dict,
        behavior_score: float | RankerResult | dict,
        semantic_score: float | RankerResult | dict,
        eligibility_score: float | RankerResult | dict,
        role_relevance_score: float | RankerResult | dict | None = None,
    ) -> EnsembleRankingResult:
        """Combine independent rankers into the final ensemble score."""

        components = {
            "skill": self._ranker_result(skill_score, "Skill ranker score."),
            "career": self._ranker_result(career_score, "Career ranker score."),
            "behavior": self._ranker_result(behavior_score, "Behavior ranker score."),
            "semantic": self._ranker_result(semantic_score, "Semantic ranker score."),
            "eligibility": self._ranker_result(eligibility_score, "Eligibility ranker score."),
        }
        if role_relevance_score is not None:
            components["role_relevance"] = self._ranker_result(
                role_relevance_score,
                "Role relevance ranker score.",
            )
        return self.ensemble_ranker.rank(components)

    def semantic_stage_score(self, fast_score: float, semantic_score: float) -> float:
        return self._finalize(
            self._weighted_average(
                {
                    "fast": (fast_score, self.config.semantic_stage_fast_weight),
                    "semantic": (semantic_score, self.config.semantic_stage_semantic_weight),
                }
            )
        )

    def apply_penalties(
        self,
        score: float,
        *,
        honeypot_penalty: float | None = None,
        disqualifier_penalty: float | None = None,
        profile_integrity_penalty: float | None = None,
    ) -> float:
        honeypot_value = self.config.score_floor if honeypot_penalty is None else honeypot_penalty
        disqualifier_value = self.config.score_floor if disqualifier_penalty is None else disqualifier_penalty
        integrity_value = self.config.score_floor if profile_integrity_penalty is None else profile_integrity_penalty
        return score - (
            honeypot_value * self.config.honeypot_penalty_weight
            + disqualifier_value * self.config.disqualifier_penalty_weight
            + integrity_value * self.config.profile_integrity_penalty_weight
        )

    def apply_profile_integrity_penalty(self, score: float, integrity_score: float) -> float:
        return self._finalize(
            self.apply_penalties(
                score,
                profile_integrity_penalty=1.0 - self._clamp(integrity_score),
            )
        )

    def score(
        self,
        skill_score: float,
        semantic_score: float,
        career: CareerAnalysisResult | dict[str, float],
        behavior_score: float | None = None,
    ) -> float:
        career_score = career.score if isinstance(career, CareerAnalysisResult) else self._legacy_career_score(career)
        fast = self.fast_score(
            skill_score,
            career_score,
            filter_score=self.config.score_ceiling,
            behavior_score=behavior_score,
        )
        return self.final_score(
            fast,
            semantic_score,
            reasoning_score=self.config.legacy_reasoning_confidence,
        )

    def _legacy_career_score(self, career: dict[str, float]) -> float:
        score = sum(
            career.get(signal, self.config.score_floor) * weight
            for signal, weight in self.config.legacy_career_weights.items()
        )
        return self._clamp(score / self._positive_weight_sum(self.config.legacy_career_weights.values()))

    def _ranker_result(self, value: float | RankerResult | dict, fallback_explanation: str) -> RankerResult:
        if isinstance(value, RankerResult):
            return value
        if isinstance(value, dict):
            return RankerResult(
                score=float(value.get("score", self.config.score_floor)),
                confidence=float(value.get("confidence", self.config.score_ceiling)),
                explanation=list(value.get("explanation", [fallback_explanation])),
            )
        return RankerResult(
            score=float(value),
            confidence=self.config.score_ceiling,
            explanation=[fallback_explanation],
        )

    def _weighted_average(self, components: Mapping[str, tuple[float | None, float]]) -> float:
        active = {
            name: (self._clamp(score), max(weight, self.config.score_floor))
            for name, (score, weight) in components.items()
            if score is not None
        }
        total_weight = self._positive_weight_sum(weight for _, weight in active.values())
        if total_weight <= self.config.score_floor:
            return self.config.score_floor
        weighted = sum(score * weight for score, weight in active.values())
        return weighted / total_weight

    def _finalize(self, score: float) -> float:
        return round(self._clamp(score), self.config.score_precision)

    def _clamp(self, score: float) -> float:
        return min(max(score, self.config.score_floor), self.config.score_ceiling)

    def _positive_weight_sum(self, weights) -> float:
        return sum(max(weight, self.config.score_floor) for weight in weights)
