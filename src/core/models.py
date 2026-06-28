"""Domain models shared by parsers, intelligence services, ranking, and APIs."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ==========================
# Basic Models
# ==========================

@dataclass
class Skill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int


@dataclass
class CareerEntry:
    company: str
    title: str
    start_date: str
    end_date: Optional[str]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass
class Education:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: str
    tier: str


@dataclass
class Language:
    language: str
    proficiency: str


@dataclass
class Certification:
    name: str = ""
    issuer: str = ""
    year: Optional[int] = None


# ==========================
# Profile
# ==========================

@dataclass
class Profile:
    headline: str
    summary: str
    current_title: str
    current_company: str
    current_industry: str
    current_company_size: str
    years_of_experience: float
    location: str
    country: str
    anonymized_name: str


# ==========================
# Redrob Signals
# ==========================

@dataclass
class SalaryRange:
    minimum: float
    maximum: float


@dataclass
class Signals:

    open_to_work: bool

    notice_period_days: int

    recruiter_response_rate: float

    interview_completion_rate: float

    github_activity_score: float

    profile_completeness_score: float

    offer_acceptance_rate: float

    avg_response_time_hours: float

    applications_submitted_30d: int

    profile_views_received_30d: int

    search_appearance_30d: int

    saved_by_recruiters_30d: int

    preferred_work_mode: str

    willing_to_relocate: bool

    verified_email: bool

    verified_phone: bool

    connection_count: int

    endorsements_received: int

    linkedin_connected: bool

    last_active_date: str

    signup_date: str

    expected_salary: SalaryRange

    skill_assessment_scores: Dict[str, float]


# ==========================
# Candidate
# ==========================

@dataclass
class Candidate:

    candidate_id: str

    profile: Profile

    career_history: List[CareerEntry]

    education: List[Education]

    skills: List[Skill]

    certifications: List[Certification]

    languages: List[Language]

    signals: Signals

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==========================
# DNA
# ==========================

@dataclass
class CandidateDNA:

    technical: Dict[str, float] = field(default_factory=dict)

    career: Dict[str, float] = field(default_factory=dict)

    behavior: Dict[str, float] = field(default_factory=dict)

    trust: Dict[str, float] = field(default_factory=dict)


# ==========================
# Job DNA
# ==========================

@dataclass
class JobDNA:

    role: str

    experience_min: int

    experience_max: int

    required_skills: List[str]

    preferred_skills: List[str]

    responsibilities: List[str]

    industries: List[str]

    education: List[str]

    work_mode: str

    location: str = ""

    notice_period_max_days: int = 90

    title_keywords: List[str] = field(default_factory=list)

    required_education: List[str] = field(default_factory=list)

    technical: Dict[str, float] = field(default_factory=dict)

    career: Dict[str, float] = field(default_factory=dict)

    behavior: Dict[str, float] = field(default_factory=dict)

    trust: Dict[str, float] = field(default_factory=dict)

    source_text: str = ""

    required_tools: List[str] = field(default_factory=list)

    preferred_tools: List[str] = field(default_factory=list)

    required_domains: List[str] = field(default_factory=list)

    preferred_domains: List[str] = field(default_factory=list)

    required_behaviors: List[str] = field(default_factory=list)

    required_experience: Dict[str, Any] = field(default_factory=dict)

    disqualifiers: List[str] = field(default_factory=list)

    evaluation_signals: List[str] = field(default_factory=list)

    company_culture: Dict[str, Any] = field(default_factory=dict)

    def to_candidate_dna(self) -> CandidateDNA:
        return CandidateDNA(
            technical=self.technical,
            career=self.career,
            behavior=self.behavior,
            trust=self.trust,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



# ==========================
# Decision Models
# ==========================

@dataclass
class Evidence:
    category: str
    statement: str
    score: float


@dataclass
class PanelResult:
    score: float
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class Decision:
    candidate_id: str
    final_score: float
    confidence: float
    panel_results: Dict[str, PanelResult]


@dataclass(frozen=True)
class EligibilityResult:
    """Hard-filter outcome used before expensive semantic ranking."""

    passed: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SkillMatchResult:
    """Weighted skill-match output with matched and missing skill evidence."""

    score: float
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    matched_required: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    matched_preferred: List[str] = field(default_factory=list)
    missing_preferred: List[str] = field(default_factory=list)
    weighted_hits: Dict[str, float] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    explanation: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CareerAnalysisResult:
    """Career-pattern scores used to identify senior recruiter-relevant signals."""

    score: float
    signals: Dict[str, float] = field(default_factory=dict)
    relevant_experience: List[str] = field(default_factory=list)
    relevant_projects: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> float:
        return self.signals[key]

    def get(self, key: str, default: float = 0.0) -> float:
        return self.signals.get(key, default)


@dataclass(frozen=True)
class RoleRelevanceResult:
    """Role alignment signal from titles, industries, progression, and transition evidence."""

    score: float
    current_title_score: float
    previous_title_score: float
    industry_score: float
    progression_score: float
    transition_score: float
    unrelated_penalty: float
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMReasoningResult:
    """Structured final reasoning contract; currently deterministic and API-ready."""

    overall_match_percent: float
    strengths: List[str]
    weaknesses: List[str]
    missing_skills: List[str]
    hiring_recommendation: str
    risk_assessment: str
    summary: str
    confidence: float
    matched_skills: List[str] = field(default_factory=list)
    relevant_experience: List[str] = field(default_factory=list)
    behavior_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfileIntegrityResult:
    """Suspicious-profile analysis used to penalize low-integrity matches."""

    integrity_score: float
    suspicious: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankedCandidate:
    """End-to-end candidate ranking result returned by pipeline, export, and API."""

    candidate: Candidate
    fast_score: float
    filter_score: float
    skill_score: float
    career_score: float
    semantic_score: float
    llm_confidence: float
    final_score: float
    recommendation: str
    why_selected: List[str]
    why_rejected: List[str]
    matched_skills: List[str]
    missing_skills: List[str]
    relevant_experience: List[str]
    relevant_projects: List[str]
    confidence: float
    reasoning: Dict[str, Any]
    behavior_score: float = 0.0
    behavior: Dict[str, Any] = field(default_factory=dict)
    profile_integrity: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["candidate"] = self.candidate.to_dict()
        return payload
