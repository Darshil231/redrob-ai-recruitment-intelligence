import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.core.config import AppConfig, RankingConfig, SemanticConfig
from src.core.models import RankedCandidate
from src.export.csv_export import CSVExporter
from src.export.evaluation_dashboard import EvaluationDashboard
from src.export.json_export import JSONExporter
from src.export.submission_generator import SubmissionGenerator
from src.intelligence.career_analyzer import CareerAnalyzer
from src.intelligence.eligibility_filter import EligibilityFilter
from src.intelligence.job_dna import JobDNAExtractor
from src.intelligence.llm_reasoner import LLMReasoner
from src.intelligence.profile_integrity_analyzer import ProfileIntegrityAnalyzer
from src.intelligence.role_relevance_analyzer import RoleRelevanceAnalyzer
from src.intelligence.semantic_matcher import SemanticMatcher
from src.intelligence.skill_mapper import SkillMapper
from src.parsers.candidate_parser import CandidateParser
from src.ranking.feature_importance import FeatureImportance
from src.ranking.pipeline import RankingPipeline
from src.ranking.rankers import EnsembleRanker, RankerResult
from src.ranking.ranking_engine import RankingEngine
from src.ranking.weight_optimizer import WeightOptimizer
from validate_submission import SubmissionValidationError, validate_submission


def suspicious_profile(candidate):
    suspicious = deepcopy(candidate)
    suspicious.profile.current_title = "Staff AI Engineer"
    suspicious.profile.current_company = "Different Co"
    suspicious.profile.summary = " ".join(["Python LLM RAG embeddings vector database ranking"] * 8)
    suspicious.career_history[0].company = "Legacy Services"
    suspicious.career_history[0].title = "Operations Analyst"
    suspicious.career_history[0].industry = "Logistics"
    suspicious.career_history[0].description = "Managed vendor operations and monthly reporting."
    suspicious.skills = suspicious.skills + [
        type(suspicious.skills[0])("TensorFlow", "beginner", 1, 1),
        type(suspicious.skills[0])("PyTorch", "beginner", 1, 1),
        type(suspicious.skills[0])("Kafka", "beginner", 1, 1),
        type(suspicious.skills[0])("Airflow", "beginner", 1, 1),
        type(suspicious.skills[0])("Kubernetes", "beginner", 1, 1),
        type(suspicious.skills[0])("Qdrant", "beginner", 1, 1),
        type(suspicious.skills[0])("Growth Marketing", "beginner", 1, 1),
        type(suspicious.skills[0])("Payroll", "beginner", 1, 1),
        type(suspicious.skills[0])("Photoshop", "beginner", 1, 1),
        type(suspicious.skills[0])("Cold Calling", "beginner", 1, 1),
        type(suspicious.skills[0])("Event Planning", "beginner", 1, 1),
        type(suspicious.skills[0])("Procurement", "beginner", 1, 1),
        type(suspicious.skills[0])("Inventory", "beginner", 1, 1),
        type(suspicious.skills[0])("Legal Drafting", "beginner", 1, 1),
    ]
    suspicious.signals.profile_completeness_score = 0.99
    suspicious.signals.verified_email = False
    suspicious.signals.verified_phone = False
    suspicious.signals.applications_submitted_30d = 100
    suspicious.signals.recruiter_response_rate = 0.1
    return suspicious


def test_candidate_parser_builds_candidate(tmp_path, candidate):
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps(candidate.to_dict()) + "\n", encoding="utf-8")

    parsed = list(CandidateParser(str(path)).parse())

    assert parsed[0].candidate_id == "c-1"
    assert parsed[0].profile.current_title == "Senior Machine Learning Engineer"
    assert parsed[0].skills[0].name == "Python"


def test_job_dna_extracts_requirements(jd_text):
    dna = JobDNAExtractor().extract(jd_text)

    assert dna.experience_min == 5
    assert "python" in dna.required_skills
    assert "llm" in dna.required_skills
    assert dna.location == "Bengaluru"


def test_job_dna_extracts_rich_structured_intelligence():
    jd = """
    Role: Staff AI Engineer
    Location: Bengaluru
    Required qualifications:
    5-9 years building production retrieval systems with Python, embeddings,
    retrieval, ranking, vector databases, production ML, evaluation frameworks,
    and hybrid search. You have deployed models to production and still write
    hands-on code.
    Preferred qualifications:
    LoRA, QLoRA, PEFT, Learning-to-Rank, HR Tech, marketplace, distributed
    systems, large scale inference, and open source contributions.
    Behaviors:
    Ownership, product mindset, startup mentality, fast shipping, mentoring,
    and collaboration.
    Culture:
    Startup product engineering team with ownership and fast iteration.
    Evaluation:
    NDCG, MAP, MRR, A/B testing, and recruiter feedback.
    Disqualifiers:
    Research only, no production deployment, only recent LLM experience,
    or no coding in last 18 months.
    """

    dna = JobDNAExtractor().extract(jd)

    assert set(dna.required_skills) >= {
        "python",
        "embeddings",
        "retrieval",
        "ranking",
        "vector_db",
        "production_ml",
        "evaluation_frameworks",
        "hybrid_search",
    }
    assert "lora" not in dna.required_skills
    assert set(dna.preferred_skills) >= {
        "lora",
        "qlora",
        "peft",
        "learning_to_rank",
        "hr_tech",
        "marketplace",
        "distributed_systems",
        "large_scale_inference",
        "open_source",
    }
    assert dna.required_experience["years"] == {"min": 5, "max": 9, "label": "5-9 years"}
    assert dna.required_experience["production_deployment"] is True
    assert dna.required_experience["recent_coding"] is True
    assert dna.required_experience["production_retrieval_systems"] is True
    assert set(dna.required_behaviors) >= {"ownership", "product_mindset", "fast_shipping", "mentoring", "collaboration"}
    assert set(dna.disqualifiers) == {
        "research_only",
        "no_production_deployment",
        "only_recent_llm_experience",
        "no_coding_last_18_months",
    }
    assert set(dna.evaluation_signals) == {"ndcg", "map", "mrr", "ab_testing", "recruiter_feedback"}
    assert set(dna.company_culture) == {"startup", "ownership", "product_engineering", "fast_iteration"}


def test_eligibility_filter_passes_strong_candidate(candidate, jd_text):
    dna = JobDNAExtractor().extract(jd_text)
    result = EligibilityFilter().evaluate(candidate, dna)

    assert result.passed is True
    assert result.score >= 0.7
    assert not any(reason.startswith("Hard fail") for reason in result.rejection_reasons)


def test_skill_mapper_weights_required_and_preferred_skills(candidate):
    dna = JobDNAExtractor().extract(
        """
        Role: Senior AI Engineer
        Required skills: Python, LLM, RAG, FastAPI.
        Preferred skills: LoRA, Qdrant.
        """
    )

    result = SkillMapper().evaluate(candidate, dna)

    assert result.score >= 0.75
    assert set(result.matched_required) >= {"python", "llm", "rag", "fastapi"}
    assert set(result.missing_preferred) == {"lora", "qdrant"}
    assert result.matched_skills == result.matched_required
    assert result.missing_skills == result.missing_required
    assert any("Required skills contribute 80%" in item for item in result.explanation)


def test_skill_mapper_normalizes_aliases_and_supports_fuzzy_matching(candidate):
    candidate.skills[2].name = "Retrieval Augmented Generation"
    candidate.skills[3].name = "SentenceTransformers"
    dna = JobDNAExtractor().extract(
        """
        Required skills: Python, RAG, Sentence Transformers.
        Preferred skills: HuggingFace, OpenSearch.
        """
    )

    result = SkillMapper().evaluate(candidate, dna)

    assert {"python", "rag", "sentence_transformers"} <= set(result.matched_required)
    assert "huggingface" in result.matched_preferred
    assert "opensearch" in result.missing_preferred


def test_skill_mapper_penalizes_keyword_stuffing(candidate):
    stuffing = " ".join(["Python LLM RAG embeddings vector database ranking"] * 8)
    candidate.profile.years_of_experience = 1.0
    candidate.profile.summary = stuffing
    candidate.skills = candidate.skills + [
        type(candidate.skills[0])("LoRA", "beginner", 0, 1),
        type(candidate.skills[0])("QLoRA", "beginner", 0, 1),
        type(candidate.skills[0])("PEFT", "beginner", 0, 1),
        type(candidate.skills[0])("Milvus", "beginner", 0, 1),
        type(candidate.skills[0])("Qdrant", "beginner", 0, 1),
        type(candidate.skills[0])("Weaviate", "beginner", 0, 1),
        type(candidate.skills[0])("Pinecone", "beginner", 0, 1),
        type(candidate.skills[0])("Elasticsearch", "beginner", 0, 1),
        type(candidate.skills[0])("OpenSearch", "beginner", 0, 1),
        type(candidate.skills[0])("Spark", "beginner", 0, 1),
        type(candidate.skills[0])("Kafka", "beginner", 0, 1),
        type(candidate.skills[0])("Airflow", "beginner", 0, 1),
        type(candidate.skills[0])("PyTorch", "beginner", 0, 1),
        type(candidate.skills[0])("TensorFlow", "beginner", 0, 1),
        type(candidate.skills[0])("LangChain", "beginner", 0, 1),
    ]
    dna = JobDNAExtractor().extract(
        """
        Required skills: Python, LLM, RAG, embeddings, retrieval, ranking.
        Preferred skills: LoRA, QLoRA, PEFT, Milvus, Qdrant, Weaviate,
        Pinecone, Elasticsearch, OpenSearch, Spark, Kafka, Airflow, PyTorch,
        TensorFlow, LangChain.
        """
    )

    result = SkillMapper().evaluate(candidate, dna)

    assert result.score < 1.0
    assert any("keyword-stuffing penalty" in item for item in result.explanation)


def test_profile_integrity_analyzer_detects_suspicious_profiles(candidate):
    result = ProfileIntegrityAnalyzer().analyze(suspicious_profile(candidate))

    assert result.suspicious is True
    assert result.integrity_score < 0.70
    assert any("Keyword stuffing" in reason for reason in result.reasons)
    assert any("AI skills" in reason for reason in result.reasons)
    assert any("current company" in reason for reason in result.reasons)
    assert any("application volume" in reason for reason in result.reasons)


def test_role_relevance_rewards_ai_roles_and_penalizes_unrelated_professions(candidate, jd_text):
    dna = JobDNAExtractor().extract(jd_text)
    analyzer = RoleRelevanceAnalyzer()

    strong = analyzer.analyze(candidate, dna)
    unrelated = deepcopy(candidate)
    unrelated.profile.current_title = "Civil Engineer"
    unrelated.profile.current_industry = "Construction"
    unrelated.profile.headline = "Civil Engineer"
    unrelated.profile.summary = "Designed roads, bridges, and site plans for construction projects."
    unrelated.career_history[0].title = "Civil Engineer"
    unrelated.career_history[0].industry = "Construction"
    unrelated.career_history[0].description = "Managed structural drawings and site execution."
    unrelated.skills = [
        type(candidate.skills[0])("AutoCAD", "advanced", 12, 60),
        type(candidate.skills[0])("Project Planning", "advanced", 8, 48),
    ]
    unrelated_result = analyzer.analyze(unrelated, dna)

    transition = deepcopy(unrelated)
    transition.profile.summary = "Civil engineer transitioning into machine learning with Python, LLM, RAG, embeddings, and model serving projects."
    transition.skills.append(type(candidate.skills[0])("Python", "advanced", 10, 24))
    transition.skills.append(type(candidate.skills[0])("RAG", "intermediate", 6, 12))
    transition_result = analyzer.analyze(transition, dna)

    assert strong.score >= 0.80
    assert unrelated_result.score <= 0.20
    assert transition_result.score > unrelated_result.score
    assert any("Unrelated profession penalty" in reason for reason in unrelated_result.reasoning)


def test_llm_reasoner_generates_recruiter_summary_under_150_words(candidate, jd_text):
    job_dna = JobDNAExtractor().extract(jd_text)
    skill = SkillMapper().evaluate(candidate, job_dna)
    career = CareerAnalyzer().analyze(candidate)

    result = LLMReasoner().reason(
        candidate=candidate,
        job_dna=job_dna,
        semantic_score=0.82,
        skill_match=skill,
        career=career,
        filter_score=0.95,
    )

    for heading in (
        "Overall Match:",
        "Matched Skills:",
        "Missing Skills:",
        "Relevant Experience:",
        "Behavior Summary:",
        "Strengths:",
        "Weaknesses:",
        "Hiring Recommendation:",
        "Confidence:",
    ):
        assert heading in result.summary
    assert len(result.summary.split()) <= 150
    assert result.matched_skills
    assert result.behavior_summary


def test_ranking_engine_uses_required_formula():
    engine = RankingEngine(RankingConfig())

    fast = engine.fast_score(skill_score=0.8, career_score=0.6, filter_score=1.0, behavior_score=0.9)
    final = engine.final_score(fast_score=fast, semantic_score=0.7, llm_confidence=0.9)

    assert fast == 0.81
    assert final == 0.784


def test_ranking_engine_keeps_legacy_fast_score_call_compatible():
    engine = RankingEngine(RankingConfig())

    fast = engine.fast_score(skill_score=0.8, career_score=0.6, filter_score=1.0)

    assert 0.0 <= fast <= 1.0


def test_ranking_engine_uses_configurable_component_weights_and_penalties():
    engine = RankingEngine(
        RankingConfig(
            skill_weight=0.20,
            career_weight=0.10,
            behavior_weight=0.10,
            semantic_weight=0.20,
            eligibility_weight=0.20,
            reasoning_weight=0.20,
            honeypot_penalty_weight=0.50,
            disqualifier_penalty_weight=0.25,
        )
    )

    score = engine.score_components(
        skill_score=1.0,
        career_score=0.5,
        behavior_score=0.5,
        semantic_score=1.0,
        eligibility_score=1.0,
        reasoning_score=0.5,
        honeypot_penalty=0.4,
        disqualifier_penalty=0.2,
    )

    assert score == 0.55


def test_ranking_engine_uses_configurable_semantic_stage_weights():
    engine = RankingEngine(
        RankingConfig(
            semantic_stage_fast_weight=0.80,
            semantic_stage_semantic_weight=0.20,
        )
    )

    assert engine.semantic_stage_score(fast_score=1.0, semantic_score=0.0) == 0.8


def test_ensemble_ranker_combines_component_votes():
    ranker = EnsembleRanker(
        RankingConfig(
            skill_weight=0.50,
            career_weight=0.25,
            behavior_weight=0.0,
            semantic_weight=0.25,
            eligibility_weight=0.0,
        )
    )

    result = ranker.rank(
        {
            "skill": RankerResult(score=0.8, confidence=0.9, explanation=["Strong skill match."]),
            "career": RankerResult(score=0.6, confidence=0.8, explanation=["Relevant career history."]),
            "semantic": RankerResult(score=1.0, confidence=0.7, explanation=["High semantic similarity."]),
        }
    )

    assert result.final_score == 0.8
    assert result.component_scores == {"skill": 0.8, "career": 0.6, "semantic": 1.0}
    assert result.confidence == 0.825
    assert "weighted voting" in result.explanation[0]


def test_ensemble_ranker_supports_confidence_weighting():
    ranker = EnsembleRanker(
        RankingConfig(skill_weight=1.0, semantic_weight=1.0),
        confidence_weighting=True,
    )

    result = ranker.rank(
        {
            "skill": {"score": 1.0, "confidence": 0.2, "explanation": ["Low-confidence skill vote."]},
            "semantic": {"score": 0.0, "confidence": 1.0, "explanation": ["High-confidence semantic vote."]},
        },
        weights={"skill": 1.0, "semantic": 1.0},
    )

    assert result.final_score == 0.167
    assert result.confidence == 0.867
    assert "confidence-weighted voting" in result.explanation[0]


def test_feature_importance_visualizes_contribution_percentages():
    ensemble = EnsembleRanker(
        RankingConfig(
            skill_weight=0.50,
            career_weight=0.25,
            behavior_weight=0.0,
            semantic_weight=0.25,
            eligibility_weight=0.0,
            profile_integrity_penalty_weight=0.20,
        )
    ).rank(
        {
            "skill": RankerResult(score=0.8, confidence=0.9, explanation=[]),
            "career": RankerResult(score=0.6, confidence=0.8, explanation=[]),
            "semantic": RankerResult(score=1.0, confidence=0.7, explanation=[]),
        }
    )

    result = FeatureImportance(
        RankingConfig(
            skill_weight=0.50,
            career_weight=0.25,
            behavior_weight=0.0,
            semantic_weight=0.25,
            eligibility_weight=0.0,
            profile_integrity_penalty_weight=0.20,
        )
    ).analyze(ensemble, {"integrity_score": 0.5})

    payload = result.to_dict()
    features = {item["feature"]: item for item in payload["features"]}
    assert {"Skill", "Career", "Behavior", "Role Relevance", "Semantic", "Eligibility", "Integrity"} == set(features)
    assert features["Skill"]["percentage"] > features["Career"]["percentage"]
    assert features["Integrity"]["percentage"] < 0
    assert "|" in features["Skill"]["visualization"]
    assert "Feature contribution mix" in payload["explanation"]


def test_weight_optimizer_uses_ground_truth_and_stores_config(tmp_path):
    samples = [
        {
            "candidate_id": "best",
            "skill_score": 0.1,
            "career_score": 0.1,
            "behavior_score": 1.0,
            "semantic_score": 0.1,
            "reasoning_score": 0.1,
        },
        {
            "candidate_id": "skill_only",
            "skill_score": 1.0,
            "career_score": 0.1,
            "behavior_score": 0.1,
            "semantic_score": 0.1,
            "reasoning_score": 0.1,
        },
        {
            "candidate_id": "career_only",
            "skill_score": 0.1,
            "career_score": 1.0,
            "behavior_score": 0.1,
            "semantic_score": 0.1,
            "reasoning_score": 0.1,
        },
    ]

    result = WeightOptimizer(grid_step=0.5, store_path=tmp_path / "weights.json").optimize(
        samples,
        ground_truth={"best": 3.0, "skill_only": 1.0, "career_only": 0.0},
    )

    assert result.objective == "ndcg"
    assert result.best_weights["behavior"] > result.best_weights["career"]
    assert result.config.behavior_weight == result.best_weights["behavior"]
    assert (tmp_path / "weights.json").exists()


def test_weight_optimizer_supports_missing_ground_truth():
    samples = [
        {"candidate_id": "a", "skill_score": 0.9, "career_score": 0.7, "behavior_score": 0.8, "semantic_score": 0.9},
        {"candidate_id": "b", "skill_score": 0.2, "career_score": 0.3, "behavior_score": 0.4, "semantic_score": 0.1},
    ]

    result = WeightOptimizer(grid_step=0.5).optimize(samples, store=False)

    assert result.objective == "unsupervised_margin"
    assert set(result.best_weights) == {"skill", "career", "behavior", "semantic", "reasoning"}
    assert round(sum(result.best_weights.values()), 3) == 1.0


def test_semantic_matcher_reuses_fake_embeddings(candidate, monkeypatch, tmp_path):
    matcher = SemanticMatcher(SemanticConfig(cache_dir=tmp_path))

    monkeypatch.setattr(matcher, "_embed_jd", lambda jd: np.array([1.0, 0.0]))
    monkeypatch.setattr(matcher, "_embed_candidates", lambda candidates: np.array([[0.8, 0.6]]))

    scores = matcher.batch_similarity([candidate], "python llm role")

    assert scores.shape == (1,)
    assert round(float(scores[0]), 2) == 0.8


def test_pipeline_returns_explainable_results(candidate, jd_text, monkeypatch):
    pipeline = RankingPipeline(
        config=AppConfig(ranking=RankingConfig(fast_top_n=10, semantic_top_n=5, final_top_n=3))
    )
    monkeypatch.setattr(pipeline.semantic_matcher, "batch_similarity", lambda candidates, jd: np.array([0.82]))

    results = pipeline.rank(jd_text, [candidate])

    assert len(results) == 1
    assert results[0].final_score > 0
    assert results[0].behavior_score > 0
    assert results[0].behavior["behavior_score"] == results[0].behavior_score
    assert "behavior" in results[0].reasoning
    assert any(reason.startswith("Behavior score is") for reason in results[0].why_selected)
    assert results[0].matched_skills
    assert results[0].reasoning["hiring_recommendation"]
    assert "ensemble" in results[0].reasoning
    assert set(results[0].reasoning["ensemble"]["component_scores"]) == {
        "skill",
        "career",
        "behavior",
        "role_relevance",
        "semantic",
        "eligibility",
    }
    assert "role_relevance" in results[0].reasoning
    assert any(reason.startswith("Role relevance score is") for reason in results[0].why_selected)
    assert "feature_importance" in results[0].reasoning
    assert results[0].feature_importance["features"]
    assert any("Feature contribution mix" in reason for reason in results[0].why_selected)


def test_pipeline_penalizes_suspicious_profile_integrity(candidate, jd_text, monkeypatch):
    pipeline = RankingPipeline(
        config=AppConfig(ranking=RankingConfig(fast_top_n=10, semantic_top_n=5, final_top_n=3))
    )
    monkeypatch.setattr(pipeline.semantic_matcher, "batch_similarity", lambda candidates, jd: np.array([0.82]))

    result = pipeline.rank(jd_text, [suspicious_profile(candidate)])[0]
    ensemble_score = result.reasoning["ensemble"]["final_score"]

    assert result.profile_integrity["suspicious"] is True
    assert result.final_score < ensemble_score
    assert "profile_integrity" in result.reasoning
    assert any("Keyword stuffing" in reason for reason in result.why_rejected)


def test_exports_include_behavior_scores(candidate, jd_text, monkeypatch, tmp_path):
    pipeline = RankingPipeline(
        config=AppConfig(ranking=RankingConfig(fast_top_n=10, semantic_top_n=5, final_top_n=3))
    )
    monkeypatch.setattr(pipeline.semantic_matcher, "batch_similarity", lambda candidates, jd: np.array([0.82]))
    results = pipeline.rank(jd_text, [candidate])

    csv_path = CSVExporter().export(results, tmp_path / "ranked.csv")
    json_path = JSONExporter().export(results, tmp_path / "ranked.json")

    csv_text = csv_path.read_text(encoding="utf-8")
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "Behavior Score" in csv_text
    assert "behavior_score" in json_payload[0]
    assert "behavior" in json_payload[0]["reasoning"]


def test_submission_generator_writes_exactly_100_sorted_rows(candidate, tmp_path):
    results = []
    for index in range(105):
        candidate_copy = replace(candidate)
        candidate_copy.candidate_id = f"c-{index:03d}"
        results.append(
            RankedCandidate(
                candidate=candidate_copy,
                fast_score=0.8,
                filter_score=1.0,
                skill_score=0.8,
                career_score=0.7,
                semantic_score=0.8,
                llm_confidence=0.8,
                final_score=1.0 if index in (2, 1) else round(0.99 - index / 1000, 6),
                recommendation="Worth recruiter screen.",
                why_selected=["Matches required skills.", "Relevant experience."],
                why_rejected=[],
                matched_skills=["python"],
                missing_skills=[],
                relevant_experience=[],
                relevant_projects=[],
                confidence=0.8,
                reasoning={"summary": "Synthetic ranking result."},
            )
        )

    path = SubmissionGenerator().generate(results, tmp_path / "submission.csv")
    rows = list(__import__("csv").DictReader(path.open("r", encoding="utf-8")))

    assert validate_submission(path) is True
    assert len(rows) == 100
    assert rows[0]["candidate_id"] == "c-001"
    assert rows[1]["candidate_id"] == "c-002"
    assert rows[2]["candidate_id"] == "c-000"
    assert set(rows[0].keys()) == {"candidate_id", "rank", "score", "reasoning"}


def test_submission_validator_fails_bad_submission(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("candidate_id,rank,score,reasoning\nc-1,1,0.5,\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError):
        validate_submission(path)


def test_evaluation_dashboard_displays_run_metrics(candidate, tmp_path):
    result = RankedCandidate(
        candidate=candidate,
        fast_score=0.8,
        filter_score=1.0,
        skill_score=0.8,
        career_score=0.7,
        semantic_score=0.8,
        llm_confidence=0.8,
        final_score=0.82,
        recommendation="Worth recruiter screen.",
        why_selected=["Matches required skills."],
        why_rejected=[],
        matched_skills=["python", "rag"],
        missing_skills=[],
        relevant_experience=[],
        relevant_projects=[],
        confidence=0.8,
        reasoning={"summary": "Synthetic ranking result."},
        behavior_score=0.76,
    )
    metrics = {
        "candidates_parsed": 120,
        "filtered": 80,
        "rejected": 40,
        "semantic_ranked": 50,
        "average_score": 0.82,
        "execution_time_seconds": 1.25,
        "memory_usage_mb": 12.5,
    }

    path = EvaluationDashboard().export(
        [result],
        metrics,
        tmp_path / "evaluation_dashboard.html",
        csv_generated=True,
        validation_passed=True,
    )

    html = path.read_text(encoding="utf-8")
    for label in (
        "Candidates Parsed",
        "Filtered",
        "Rejected",
        "Semantic Ranked",
        "Average Score",
        "Top Skills",
        "Behavior Distribution",
        "Execution Time",
        "Memory Usage",
        "CSV Generated",
        "Validation Passed",
    ):
        assert label in html
    assert "python" in html
    assert "High" in html
