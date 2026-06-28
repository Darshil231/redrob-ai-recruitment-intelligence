import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.core.models import (
    Candidate,
    CareerEntry,
    Certification,
    Education,
    Language,
    Profile,
    SalaryRange,
    Signals,
    Skill,
)


@pytest.fixture
def candidate() -> Candidate:
    return Candidate(
        candidate_id="c-1",
        profile=Profile(
            headline="Senior ML Engineer",
            summary="Built production LLM and RAG systems with FastAPI.",
            current_title="Senior Machine Learning Engineer",
            current_company="Acme AI",
            current_industry="AI",
            current_company_size="startup",
            years_of_experience=7.0,
            location="Bengaluru",
            country="India",
            anonymized_name="Candidate One",
        ),
        career_history=[
            CareerEntry(
                company="Acme AI",
                title="Senior ML Engineer",
                start_date="2021",
                end_date=None,
                duration_months=42,
                is_current=True,
                industry="AI",
                company_size="startup",
                description="Led deployed LLM, RAG, embeddings and FastAPI services on AWS.",
            )
        ],
        education=[
            Education(
                institution="Example University",
                degree="Bachelor",
                field_of_study="Computer Science",
                start_year=2012,
                end_year=2016,
                grade="A",
                tier="1",
            )
        ],
        skills=[
            Skill("Python", "advanced", 25, 72),
            Skill("LLM", "advanced", 20, 36),
            Skill("RAG", "advanced", 12, 24),
            Skill("FastAPI", "advanced", 10, 30),
        ],
        certifications=[Certification()],
        languages=[Language("English", "fluent")],
        signals=Signals(
            open_to_work=True,
            notice_period_days=30,
            recruiter_response_rate=0.8,
            interview_completion_rate=0.9,
            github_activity_score=0.7,
            profile_completeness_score=0.95,
            offer_acceptance_rate=0.8,
            avg_response_time_hours=4,
            applications_submitted_30d=2,
            profile_views_received_30d=30,
            search_appearance_30d=100,
            saved_by_recruiters_30d=4,
            preferred_work_mode="hybrid",
            willing_to_relocate=True,
            verified_email=True,
            verified_phone=True,
            connection_count=500,
            endorsements_received=100,
            linkedin_connected=True,
            last_active_date="2026-06-01",
            signup_date="2024-01-01",
            expected_salary=SalaryRange(20, 35),
            skill_assessment_scores={"Python": 92},
        ),
    )


@pytest.fixture
def jd_text() -> str:
    return """
    Role: Senior Machine Learning Engineer
    Location: Bengaluru
    5 to 9 years experience.
    Required skills: Python, LLM, RAG, embeddings, FastAPI, AWS.
    Bachelor degree in Computer Science preferred.
    Build production ML systems and mentor engineers.
    """
