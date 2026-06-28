"""End-to-end recruitment ranking pipeline orchestration."""

from collections.abc import Sequence
from heapq import nlargest

from src.core.config import AppConfig
from src.core.models import Candidate, JobDNA, RankedCandidate
from src.intelligence.behavior_analyzer import BehaviorAnalyzer
from src.intelligence.career_analyzer import CareerAnalyzer
from src.intelligence.eligibility_filter import EligibilityFilter
from src.intelligence.job_dna import JobDNAExtractor
from src.intelligence.llm_reasoner import LLMReasoner
from src.intelligence.profile_integrity_analyzer import ProfileIntegrityAnalyzer
from src.intelligence.role_relevance_analyzer import RoleRelevanceAnalyzer
from src.intelligence.semantic_matcher import SemanticMatcher
from src.intelligence.skill_mapper import SkillMapper
from src.ranking.feature_importance import FeatureImportance
from src.ranking.rankers import BehaviorRanker, CareerRanker, EligibilityRanker, RoleRelevanceRanker, SemanticRanker, SkillRanker
from src.ranking.ranking_engine import RankingEngine
from src.utils.profiler import StageProfiler


class RankingPipeline:
    """Coordinates filter, fast ranking, semantic ranking, and final reasoning."""

    def __init__(
        self,
        config: AppConfig | None = None,
        dna_extractor: JobDNAExtractor | None = None,
        eligibility_filter: EligibilityFilter | None = None,
        skill_mapper: SkillMapper | None = None,
        career_analyzer: CareerAnalyzer | None = None,
        behavior_analyzer: BehaviorAnalyzer | None = None,
        role_relevance_analyzer: RoleRelevanceAnalyzer | None = None,
        profile_integrity_analyzer: ProfileIntegrityAnalyzer | None = None,
        semantic_matcher: SemanticMatcher | None = None,
        reasoner: LLMReasoner | None = None,
        ranking_engine: RankingEngine | None = None,
        profiler: StageProfiler | None = None,
    ):
        self.config = config or AppConfig()
        self.profiler = profiler
        self.skill_mapper = skill_mapper or SkillMapper()
        self.dna_extractor = dna_extractor or JobDNAExtractor(self.skill_mapper.taxonomy)
        self.eligibility_filter = eligibility_filter or EligibilityFilter(self.skill_mapper)
        self.career_analyzer = career_analyzer or CareerAnalyzer()
        self.behavior_analyzer = behavior_analyzer or BehaviorAnalyzer(self.config.behavior)
        self.role_relevance_analyzer = role_relevance_analyzer or RoleRelevanceAnalyzer()
        self.profile_integrity_analyzer = profile_integrity_analyzer or ProfileIntegrityAnalyzer(
            self.config.ranking,
            self.skill_mapper.taxonomy,
        )
        self.semantic_matcher = semantic_matcher or SemanticMatcher(self.config.semantic)
        self.reasoner = reasoner or LLMReasoner()
        self.ranking_engine = ranking_engine or RankingEngine(self.config.ranking)
        self.feature_importance = FeatureImportance(self.config.ranking)
        self.skill_ranker = SkillRanker(self.skill_mapper)
        self.career_ranker = CareerRanker(self.career_analyzer)
        self.behavior_ranker = BehaviorRanker(self.behavior_analyzer)
        self.role_relevance_ranker = RoleRelevanceRanker(self.role_relevance_analyzer)
        self.semantic_ranker = SemanticRanker(self.semantic_matcher)
        self.eligibility_ranker = EligibilityRanker(self.eligibility_filter)
        self.latest_metrics: dict[str, object] = {}

    def rank(self, jd_text: str, candidates: Sequence[Candidate]) -> list[RankedCandidate]:
        with self._measure("Job DNA extraction"):
            job_dna = self.dna_extractor.extract(jd_text)
        return self.rank_with_dna(job_dna, candidates)

    def rank_with_dna(self, job_dna: JobDNA, candidates: Sequence[Candidate]) -> list[RankedCandidate]:
        candidates_parsed = len(candidates)
        fast_stage = []
        filtered_count = 0
        rejected_count = 0
        with self._measure("Fast candidate scoring"):
            for candidate in candidates:
                with self._measure("Skill mapping"):
                    skill = self.skill_mapper.evaluate(candidate, job_dna)
                with self._measure("Eligibility filtering"):
                    eligibility = self.eligibility_filter.evaluate(candidate, job_dna, skill)
                if not eligibility.passed:
                    rejected_count += 1
                    continue
                with self._measure("Career analysis"):
                    career = self.career_analyzer.analyze(candidate)
                with self._measure("Behavior analysis"):
                    behavior = self.behavior_analyzer.analyze(candidate)
                with self._measure("Role relevance analysis"):
                    role_relevance = self.role_relevance_analyzer.analyze(candidate, job_dna)
                with self._measure("Fast ranking"):
                    fast_score = self.ranking_engine.fast_score(
                        skill.score,
                        career.score,
                        eligibility.score,
                        behavior["behavior_score"],
                        role_relevance.score,
                    )
                filtered_count += 1
                fast_stage.append((fast_score, candidate, eligibility, skill, career, behavior, role_relevance))

        with self._measure("Fast ranking"):
            fast_stage = nlargest(self.config.ranking.fast_top_n, fast_stage, key=lambda item: item[0])

        semantic_candidates = [item[1] for item in fast_stage]
        with self._measure("Semantic embedding"):
            semantic_scores = self.semantic_matcher.batch_similarity(
                semantic_candidates,
                job_dna.source_text,
            )

        with self._measure("Semantic ranking"):
            semantic_stage = [
                (*item, float(semantic_score))
                for item, semantic_score in zip(fast_stage, semantic_scores)
            ]
            semantic_stage = nlargest(
                self.config.ranking.semantic_top_n,
                semantic_stage,
                key=lambda item: self.ranking_engine.semantic_stage_score(item[0], item[-1]),
            )
        semantic_ranked_count = len(semantic_stage)

        final_stage = []
        for fast_score, candidate, eligibility, skill, career, behavior, role_relevance, semantic_score in semantic_stage:
            with self._measure("Profile integrity analysis"):
                profile_integrity = self.profile_integrity_analyzer.analyze(candidate)
            ensemble = self.ranking_engine.ensemble_score(
                skill_score=self.skill_ranker.rank(candidate, job_dna, skill_match=skill),
                career_score=self.career_ranker.rank(candidate, job_dna, career=career),
                behavior_score=self.behavior_ranker.rank(candidate, job_dna, behavior=behavior),
                role_relevance_score=self.role_relevance_ranker.rank(
                    candidate,
                    job_dna,
                    role_relevance=role_relevance,
                ),
                semantic_score=self.semantic_ranker.rank(candidate, job_dna, semantic_score=semantic_score),
                eligibility_score=self.eligibility_ranker.rank(candidate, job_dna, eligibility=eligibility),
            )
            final_score = self.ranking_engine.apply_profile_integrity_penalty(
                ensemble.final_score,
                profile_integrity.integrity_score,
            )
            feature_importance = self.feature_importance.analyze(ensemble, profile_integrity)
            final_stage.append(
                (
                    final_score,
                    fast_score,
                    candidate,
                    eligibility,
                    skill,
                    career,
                    behavior,
                    role_relevance,
                    profile_integrity,
                    semantic_score,
                    ensemble,
                    feature_importance,
                )
            )
        final_stage = nlargest(
            self.config.ranking.final_top_n,
            final_stage,
            key=lambda item: item[0],
        )

        ranked = []
        for (
            final_score,
            fast_score,
            candidate,
            eligibility,
            skill,
            career,
            behavior,
            role_relevance,
            profile_integrity,
            semantic_score,
            ensemble,
            feature_importance,
        ) in final_stage:
            with self._measure("LLM reasoning"):
                reasoning = self.reasoner.reason(
                    candidate=candidate,
                    job_dna=job_dna,
                    semantic_score=semantic_score,
                    skill_match=skill,
                    career=career,
                    filter_score=eligibility.score,
                )
            behavior_explanation = self._behavior_explanation(behavior)
            integrity_risks = profile_integrity.reasons if profile_integrity.suspicious else []
            ranked.append(
                RankedCandidate(
                    candidate=candidate,
                    fast_score=fast_score,
                    filter_score=eligibility.score,
                    skill_score=skill.score,
                    career_score=career.score,
                    behavior_score=behavior["behavior_score"],
                    semantic_score=round(semantic_score, 3),
                    llm_confidence=reasoning.confidence,
                    final_score=final_score,
                    recommendation=reasoning.hiring_recommendation,
                    why_selected=(
                        eligibility.reasons
                        + skill.reasoning
                        + career.reasoning
                        + behavior_explanation["positive"]
                        + role_relevance.reasoning
                        + ensemble.explanation[:3]
                        + [feature_importance.explanation]
                        + reasoning.strengths
                    ),
                    why_rejected=(
                        eligibility.rejection_reasons
                        + behavior_explanation["risk"]
                        + integrity_risks
                        + reasoning.weaknesses
                    ),
                    matched_skills=skill.matched_skills,
                    missing_skills=skill.missing_skills,
                    relevant_experience=career.relevant_experience,
                    relevant_projects=career.relevant_projects,
                    confidence=ensemble.confidence,
                    reasoning={
                        **reasoning.to_dict(),
                        "behavior": behavior,
                        "role_relevance": role_relevance.to_dict(),
                        "ensemble": ensemble.to_dict(),
                        "profile_integrity": profile_integrity.to_dict(),
                        "feature_importance": feature_importance.to_dict(),
                    },
                    behavior=behavior,
                    profile_integrity=profile_integrity.to_dict(),
                    feature_importance=feature_importance.to_dict(),
                )
            )

        ranked.sort(key=lambda item: item.final_score, reverse=True)
        self.latest_metrics = {
            "candidates_parsed": candidates_parsed,
            "filtered": filtered_count,
            "rejected": rejected_count,
            "semantic_ranked": semantic_ranked_count,
            "average_score": round(
                sum(result.final_score for result in ranked) / len(ranked),
                self.config.ranking.score_precision,
            )
            if ranked
            else 0.0,
        }
        return ranked

    def _measure(self, name: str):
        if self.profiler is None:
            return _NoopTimer()
        return self.profiler.measure(name)

    @staticmethod
    def _behavior_explanation(behavior: dict) -> dict[str, list[str]]:
        summary = f"Behavior score is {behavior.get('behavior_score', 0.0):.2f}."
        details = list(behavior.get("explanation", []))
        risk_terms = ("risk", "reduces", "stale", "slow", "long notice", "low offer", "high recent application")
        risk = [item for item in details if any(term in item.lower() for term in risk_terms)]
        positive = [summary] + [item for item in details if item not in risk]
        return {"positive": positive, "risk": risk}


class _NoopTimer:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
