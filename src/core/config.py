"""Central configuration for ranking thresholds, model settings, and exports."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class RankingConfig:
    fast_top_n: int = 200
    semantic_top_n: int = 50
    final_top_n: int = 10
    skill_weight: float = 0.35
    career_weight: float = 0.25
    behavior_weight: float = 0.20
    role_relevance_weight: float = 0.30
    eligibility_weight: float = 0.20
    semantic_weight: float = 0.40
    reasoning_weight: float = 0.20
    fast_score_weight: float = 0.40
    honeypot_penalty_weight: float = 0.25
    disqualifier_penalty_weight: float = 0.15
    profile_integrity_penalty_weight: float = 0.20
    profile_integrity_suspicious_threshold: float = 0.70
    semantic_stage_fast_weight: float = 0.50
    semantic_stage_semantic_weight: float = 0.50
    ensemble_confidence_weighting: bool = False
    ensemble_weighted_voting: bool = True
    legacy_reasoning_confidence: float = 0.70
    score_floor: float = 0.0
    score_ceiling: float = 1.0
    score_precision: int = 3
    legacy_career_weights: Dict[str, float] = field(default_factory=lambda: {
        "production_ml": 0.35,
        "big_data": 0.30,
        "backend": 0.20,
        "leadership": 0.15,
    })

    @property
    def filter_weight(self) -> float:
        return self.eligibility_weight

    @property
    def final_fast_weight(self) -> float:
        return self.fast_score_weight

    @property
    def final_semantic_weight(self) -> float:
        return self.semantic_weight

    @property
    def final_llm_weight(self) -> float:
        return self.reasoning_weight


@dataclass(frozen=True)
class SemanticConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 128
    cache_dir: Path = Path(".cache") / "embeddings"
    show_progress_bar: bool = False
    local_files_only: bool = True


@dataclass(frozen=True)
class BehaviorConfig:
    """Weights and normalization caps for Redrob behavioral signal scoring."""

    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "availability": 0.30,
        "engagement": 0.30,
        "trust": 0.25,
        "risk": 0.15,
    })
    availability_weights: Dict[str, float] = field(default_factory=lambda: {
        "open_to_work_flag": 0.55,
        "notice_period_days": 0.45,
    })
    engagement_weights: Dict[str, float] = field(default_factory=lambda: {
        "recruiter_response_rate": 0.24,
        "interview_completion_rate": 0.22,
        "github_activity_score": 0.16,
        "saved_by_recruiters_30d": 0.14,
        "search_appearance_30d": 0.10,
        "profile_views_received_30d": 0.10,
        "avg_response_time_hours": 0.04,
    })
    trust_weights: Dict[str, float] = field(default_factory=lambda: {
        "profile_completeness_score": 0.34,
        "verified_email": 0.18,
        "verified_phone": 0.18,
        "linkedin_connected": 0.14,
        "offer_acceptance_rate": 0.16,
    })
    risk_weights: Dict[str, float] = field(default_factory=lambda: {
        "notice_period_days": 0.20,
        "avg_response_time_hours": 0.18,
        "applications_submitted_30d": 0.20,
        "offer_acceptance_rate": 0.18,
        "inactive_user": 0.24,
    })
    notice_period_low_risk_days: int = 30
    notice_period_high_risk_days: int = 90
    response_time_good_hours: float = 4.0
    response_time_poor_hours: float = 72.0
    applications_good_30d: int = 10
    applications_excessive_30d: int = 50
    saved_by_recruiters_cap_30d: int = 10
    search_appearance_cap_30d: int = 200
    profile_views_cap_30d: int = 100
    inactive_after_days: int = 30
    stale_after_days: int = 90
    poor_offer_acceptance_rate: float = 0.50


@dataclass(frozen=True)
class AppConfig:
    candidates_path: Path = Path("data") / "candidates.jsonl"
    jd_path: Path = Path("data") / "job_description.docx"
    exports_dir: Path = Path("exports")
    ranking: RankingConfig = field(default_factory=RankingConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
