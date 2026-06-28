"""Independent rankers and ensemble score composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.core.config import RankingConfig
from src.core.models import Candidate, CareerAnalysisResult, EligibilityResult, JobDNA, RoleRelevanceResult, SkillMatchResult
from src.intelligence.behavior_analyzer import BehaviorAnalyzer
from src.intelligence.career_analyzer import CareerAnalyzer
from src.intelligence.eligibility_filter import EligibilityFilter
from src.intelligence.role_relevance_analyzer import RoleRelevanceAnalyzer
from src.intelligence.semantic_matcher import SemanticMatcher
from src.intelligence.skill_mapper import SkillMapper


@dataclass(frozen=True)
class RankerResult:
    """Standard output contract for every independent ranker."""

    score: float
    confidence: float
    explanation: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnsembleRankingResult:
    """Combined ensemble score and supporting component diagnostics."""

    final_score: float
    component_scores: dict[str, float]
    confidence: float
    explanation: list[str]
    component_results: dict[str, RankerResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_score": self.final_score,
            "component_scores": self.component_scores,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "component_results": {
                name: result.to_dict()
                for name, result in self.component_results.items()
            },
        }


class SkillRanker:
    """Scores technical fit from required and preferred skill coverage."""

    def __init__(self, skill_mapper: SkillMapper | None = None):
        self.skill_mapper = skill_mapper or SkillMapper()

    def rank(
        self,
        candidate: Candidate,
        job_dna: JobDNA,
        *,
        skill_match: SkillMatchResult | None = None,
    ) -> RankerResult:
        result = skill_match or self.skill_mapper.evaluate(candidate, job_dna)
        extracted = len(job_dna.required_skills) + len(job_dna.required_tools) + len(job_dna.preferred_skills)
        confidence = 0.90 if extracted else 0.55
        return RankerResult(
            score=result.score,
            confidence=confidence,
            explanation=list(result.explanation or result.reasoning),
        )


class CareerRanker:
    """Scores career-pattern alignment with the target role."""

    def __init__(self, career_analyzer: CareerAnalyzer | None = None):
        self.career_analyzer = career_analyzer or CareerAnalyzer()

    def rank(
        self,
        candidate: Candidate,
        job_dna: JobDNA | None = None,
        *,
        career: CareerAnalysisResult | None = None,
    ) -> RankerResult:
        del job_dna
        result = career or self.career_analyzer.analyze(candidate)
        history_depth = min(len(candidate.career_history) / 4, 1.0)
        years_depth = min(candidate.profile.years_of_experience / 10, 1.0)
        confidence = 0.55 + (history_depth * 0.25) + (years_depth * 0.15)
        return RankerResult(
            score=result.score,
            confidence=round(min(confidence, 0.95), 3),
            explanation=list(result.reasoning),
        )


class BehaviorRanker:
    """Scores marketplace behavior, availability, engagement, trust, and risk."""

    def __init__(self, behavior_analyzer: BehaviorAnalyzer | None = None):
        self.behavior_analyzer = behavior_analyzer or BehaviorAnalyzer()

    def rank(
        self,
        candidate: Candidate,
        job_dna: JobDNA | None = None,
        *,
        behavior: Mapping[str, Any] | None = None,
    ) -> RankerResult:
        del job_dna
        result = dict(behavior or self.behavior_analyzer.analyze(candidate))
        confidence = 0.70
        if result.get("trust", 0.0) >= 0.75:
            confidence += 0.15
        if result.get("engagement", 0.0) >= 0.50:
            confidence += 0.05
        return RankerResult(
            score=float(result.get("behavior_score", 0.0)),
            confidence=round(min(confidence, 0.95), 3),
            explanation=list(result.get("explanation", [])),
        )


class RoleRelevanceRanker:
    """Scores title, industry, and career trajectory relevance to the target role."""

    def __init__(self, analyzer: RoleRelevanceAnalyzer | None = None):
        self.analyzer = analyzer or RoleRelevanceAnalyzer()

    def rank(
        self,
        candidate: Candidate,
        job_dna: JobDNA,
        *,
        role_relevance: RoleRelevanceResult | None = None,
    ) -> RankerResult:
        result = role_relevance or self.analyzer.analyze(candidate, job_dna)
        confidence = 0.90
        if result.unrelated_penalty:
            confidence = 0.95
        elif result.transition_score >= 0.35:
            confidence = 0.85
        return RankerResult(
            score=result.score,
            confidence=confidence,
            explanation=list(result.reasoning),
        )


class SemanticRanker:
    """Scores semantic similarity between the candidate profile and JD text."""

    def __init__(self, semantic_matcher: SemanticMatcher | None = None):
        self.semantic_matcher = semantic_matcher

    def rank(
        self,
        candidate: Candidate,
        job_dna: JobDNA,
        *,
        semantic_score: float | None = None,
    ) -> RankerResult:
        if semantic_score is None:
            if self.semantic_matcher is None:
                raise ValueError("semantic_score is required when no SemanticMatcher is configured.")
            semantic_score = float(self.semantic_matcher.batch_similarity([candidate], job_dna.source_text)[0])
        confidence = 0.85 if job_dna.source_text else 0.65
        return RankerResult(
            score=float(semantic_score),
            confidence=confidence,
            explanation=[f"Semantic profile-to-JD similarity is {float(semantic_score):.3f}."],
        )


class EligibilityRanker:
    """Scores hard-filter compatibility and exposes rejection evidence."""

    def __init__(self, eligibility_filter: EligibilityFilter | None = None):
        self.eligibility_filter = eligibility_filter or EligibilityFilter()

    def rank(
        self,
        candidate: Candidate,
        job_dna: JobDNA,
        *,
        eligibility: EligibilityResult | None = None,
    ) -> RankerResult:
        result = eligibility or self.eligibility_filter.evaluate(candidate, job_dna)
        confidence = 0.98 if result.passed else 0.92
        return RankerResult(
            score=result.score,
            confidence=confidence,
            explanation=list(result.reasons + result.rejection_reasons),
        )


class EnsembleRanker:
    """Combines independent ranker outputs with configurable weighted voting."""

    DEFAULT_COMPONENT_WEIGHTS = {
        "skill": "skill_weight",
        "career": "career_weight",
        "behavior": "behavior_weight",
        "role_relevance": "role_relevance_weight",
        "semantic": "semantic_weight",
        "eligibility": "eligibility_weight",
    }

    def __init__(
        self,
        config: RankingConfig | None = None,
        *,
        confidence_weighting: bool | None = None,
        weighted_voting: bool | None = None,
    ):
        self.config = config or RankingConfig()
        self.confidence_weighting = (
            self.config.ensemble_confidence_weighting
            if confidence_weighting is None
            else confidence_weighting
        )
        self.weighted_voting = (
            self.config.ensemble_weighted_voting
            if weighted_voting is None
            else weighted_voting
        )

    def rank(
        self,
        results: Mapping[str, RankerResult | Mapping[str, Any]],
        weights: Mapping[str, float] | None = None,
    ) -> EnsembleRankingResult:
        normalized = {
            name: self._as_result(result)
            for name, result in results.items()
            if result is not None
        }
        component_weights = dict(weights or self._default_weights())
        active = {
            name: result
            for name, result in normalized.items()
            if component_weights.get(name, 0.0) > self.config.score_floor
        }
        if not active:
            return EnsembleRankingResult(
                final_score=self.config.score_floor,
                component_scores={},
                confidence=self.config.score_floor,
                explanation=["No active ranker results were available."],
                component_results={},
            )

        weighted_scores = []
        weighted_confidences = []
        for name, result in active.items():
            weight = max(component_weights[name], self.config.score_floor)
            if self.confidence_weighting:
                weight *= self._clamp(result.confidence)
            weighted_scores.append(self._clamp(result.score) * weight)
            weighted_confidences.append(self._clamp(result.confidence) * weight)

        total_weight = sum(
            max(component_weights[name], self.config.score_floor)
            * (self._clamp(result.confidence) if self.confidence_weighting else 1.0)
            for name, result in active.items()
        )
        final_score = self.config.score_floor if total_weight <= 0 else sum(weighted_scores) / total_weight
        confidence = self.config.score_floor if total_weight <= 0 else sum(weighted_confidences) / total_weight
        explanation = self._explanation(active, component_weights)

        return EnsembleRankingResult(
            final_score=round(self._clamp(final_score), self.config.score_precision),
            component_scores={
                name: round(self._clamp(result.score), self.config.score_precision)
                for name, result in active.items()
            },
            confidence=round(self._clamp(confidence), self.config.score_precision),
            explanation=explanation,
            component_results=active,
        )

    def _default_weights(self) -> dict[str, float]:
        return {
            component: getattr(self.config, field_name)
            for component, field_name in self.DEFAULT_COMPONENT_WEIGHTS.items()
        }

    def _explanation(
        self,
        results: Mapping[str, RankerResult],
        weights: Mapping[str, float],
    ) -> list[str]:
        voting_mode = "confidence-weighted voting" if self.confidence_weighting else "weighted voting"
        explanation = [f"Final score uses {voting_mode} across {len(results)} rankers."]
        for name, result in sorted(results.items()):
            explanation.append(
                f"{name.title()} ranker voted {result.score:.3f} "
                f"with weight {weights.get(name, 0.0):.3f} and confidence {result.confidence:.3f}."
            )
            explanation.extend(result.explanation[:2])
        return explanation

    @staticmethod
    def _as_result(result: RankerResult | Mapping[str, Any]) -> RankerResult:
        if isinstance(result, RankerResult):
            return result
        return RankerResult(
            score=float(result.get("score", 0.0)),
            confidence=float(result.get("confidence", 0.0)),
            explanation=list(result.get("explanation", [])),
        )

    def _clamp(self, score: float) -> float:
        return min(max(score, self.config.score_floor), self.config.score_ceiling)
