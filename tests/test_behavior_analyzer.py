from dataclasses import replace
from datetime import date

from src.core.config import BehaviorConfig
from src.intelligence.behavior_analyzer import BehaviorAnalyzer


def test_behavior_analyzer_scores_strong_candidate(candidate):
    analyzer = BehaviorAnalyzer(today_provider=lambda: date(2026, 6, 27))

    result = analyzer.analyze(candidate)

    assert 0.0 <= result["behavior_score"] <= 1.0
    assert result["behavior_score"] >= 0.75
    assert result["availability"] >= 0.75
    assert result["engagement"] >= 0.65
    assert result["trust"] >= 0.80
    assert result["risk"] <= 0.25
    assert "Candidate is marked open to work." in result["explanation"]


def test_behavior_analyzer_penalizes_stale_high_risk_signals(candidate):
    stale_signals = replace(
        candidate.signals,
        open_to_work=False,
        notice_period_days=120,
        recruiter_response_rate=0.1,
        interview_completion_rate=0.2,
        github_activity_score=0.0,
        profile_completeness_score=0.35,
        verified_email=False,
        verified_phone=False,
        linkedin_connected=False,
        saved_by_recruiters_30d=0,
        search_appearance_30d=2,
        profile_views_received_30d=1,
        avg_response_time_hours=96,
        applications_submitted_30d=75,
        offer_acceptance_rate=0.2,
        last_active_date="2026-01-01",
    )
    weak_candidate = replace(candidate, signals=stale_signals)
    analyzer = BehaviorAnalyzer(today_provider=lambda: date(2026, 6, 27))

    result = analyzer.analyze(weak_candidate)

    assert result["behavior_score"] < 0.35
    assert result["availability"] < 0.25
    assert result["trust"] < 0.35
    assert result["risk"] > 0.75
    assert "High recent application volume increases behavioral risk." in result["explanation"]
    assert "Recent activity is stale or missing." in result["explanation"]


def test_behavior_analyzer_accepts_api_style_signal_payload():
    analyzer = BehaviorAnalyzer(today_provider=lambda: date(2026, 6, 27))
    payload = {
        "candidate_id": "api-1",
        "redrob_signals": {
            "open_to_work_flag": True,
            "notice_period_days": 15,
            "recruiter_response_rate": 82,
            "interview_completion_rate": 0.9,
            "github_activity_score": 0.6,
            "profile_completeness_score": 0.95,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "saved_by_recruiters_30d": 8,
            "search_appearance_30d": 160,
            "profile_views_received_30d": 70,
            "avg_response_time_hours": 3,
            "applications_submitted_30d": 4,
            "offer_acceptance_rate": 0.9,
            "last_active_date": "2026-06-26",
        },
    }

    result = analyzer.analyze(payload)

    assert result["behavior_score"] > 0.85
    assert result["risk"] == 0.0


def test_behavior_weights_are_configurable(candidate):
    config = BehaviorConfig(category_weights={"availability": 1.0, "engagement": 0.0, "trust": 0.0, "risk": 0.0})
    analyzer = BehaviorAnalyzer(config=config, today_provider=lambda: date(2026, 6, 27))

    result = analyzer.analyze(candidate)

    assert result["behavior_score"] == result["availability"]
