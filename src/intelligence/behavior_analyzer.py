"""Behavioral signal scoring for Redrob candidate intelligence."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Callable, Mapping

from src.core.config import BehaviorConfig


class BehaviorAnalyzer:
    """Converts Redrob activity and trust signals into normalized behavior scores."""

    def __init__(
        self,
        config: BehaviorConfig | None = None,
        today_provider: Callable[[], date] | None = None,
    ):
        self.config = config or BehaviorConfig()
        self._today_provider = today_provider or date.today

    def analyze(self, candidate: Any) -> dict:
        signals = self._signals(candidate)
        explanation: list[str] = []

        availability = self._availability(signals, explanation)
        engagement = self._engagement(signals, explanation)
        trust = self._trust(signals, explanation)
        risk = self._risk(signals, explanation)

        behavior_score = self._weighted_average(
            {
                "availability": availability,
                "engagement": engagement,
                "trust": trust,
                "risk": 1.0 - risk,
            },
            self.config.category_weights,
        )

        return {
            "behavior_score": round(behavior_score, 3),
            "availability": round(availability, 3),
            "engagement": round(engagement, 3),
            "trust": round(trust, 3),
            "risk": round(risk, 3),
            "explanation": explanation,
        }

    def _availability(self, signals: Mapping[str, Any], explanation: list[str]) -> float:
        open_to_work = 1.0 if self._bool(signals, "open_to_work_flag", "open_to_work") else 0.0
        notice_days = self._number(signals, "notice_period_days", default=self.config.notice_period_high_risk_days)
        notice_score = 1.0 - self._linear_risk(
            notice_days,
            self.config.notice_period_low_risk_days,
            self.config.notice_period_high_risk_days,
        )

        if open_to_work:
            explanation.append("Candidate is marked open to work.")
        if notice_days <= self.config.notice_period_low_risk_days:
            explanation.append("Notice period supports near-term availability.")
        elif notice_days >= self.config.notice_period_high_risk_days:
            explanation.append("Long notice period reduces availability.")

        return self._weighted_average(
            {
                "open_to_work_flag": open_to_work,
                "notice_period_days": notice_score,
            },
            self.config.availability_weights,
        )

    def _engagement(self, signals: Mapping[str, Any], explanation: list[str]) -> float:
        response_rate = self._rate(signals, "recruiter_response_rate")
        interview_rate = self._rate(signals, "interview_completion_rate")
        github_activity = self._rate(signals, "github_activity_score")
        saved = self._cap_score(signals, "saved_by_recruiters_30d", self.config.saved_by_recruiters_cap_30d)
        appearances = self._cap_score(signals, "search_appearance_30d", self.config.search_appearance_cap_30d)
        views = self._cap_score(signals, "profile_views_received_30d", self.config.profile_views_cap_30d)
        response_time = 1.0 - self._linear_risk(
            self._number(signals, "avg_response_time_hours", default=self.config.response_time_poor_hours),
            self.config.response_time_good_hours,
            self.config.response_time_poor_hours,
        )

        if response_rate >= 0.75:
            explanation.append("Recruiter response rate is strong.")
        if interview_rate >= 0.80:
            explanation.append("Interview completion rate is strong.")
        if saved >= 0.5:
            explanation.append("Recruiter saves indicate recent market demand.")

        return self._weighted_average(
            {
                "recruiter_response_rate": response_rate,
                "interview_completion_rate": interview_rate,
                "github_activity_score": github_activity,
                "saved_by_recruiters_30d": saved,
                "search_appearance_30d": appearances,
                "profile_views_received_30d": views,
                "avg_response_time_hours": response_time,
            },
            self.config.engagement_weights,
        )

    def _trust(self, signals: Mapping[str, Any], explanation: list[str]) -> float:
        profile_complete = self._rate(signals, "profile_completeness_score")
        verified_email = 1.0 if self._bool(signals, "verified_email") else 0.0
        verified_phone = 1.0 if self._bool(signals, "verified_phone") else 0.0
        linkedin_connected = 1.0 if self._bool(signals, "linkedin_connected") else 0.0
        offer_acceptance = self._rate(signals, "offer_acceptance_rate")

        if profile_complete >= 0.85:
            explanation.append("Profile completeness is high.")
        if verified_email and verified_phone:
            explanation.append("Email and phone are verified.")
        if offer_acceptance < self.config.poor_offer_acceptance_rate:
            explanation.append("Low offer acceptance rate reduces trust.")

        return self._weighted_average(
            {
                "profile_completeness_score": profile_complete,
                "verified_email": verified_email,
                "verified_phone": verified_phone,
                "linkedin_connected": linkedin_connected,
                "offer_acceptance_rate": offer_acceptance,
            },
            self.config.trust_weights,
        )

    def _risk(self, signals: Mapping[str, Any], explanation: list[str]) -> float:
        notice_days = self._number(signals, "notice_period_days", default=self.config.notice_period_high_risk_days)
        response_hours = self._number(signals, "avg_response_time_hours", default=self.config.response_time_poor_hours)
        applications = self._number(signals, "applications_submitted_30d", default=0.0)
        offer_acceptance = self._rate(signals, "offer_acceptance_rate")
        inactive_risk = self._inactive_risk(signals)

        applications_risk = self._linear_risk(
            applications,
            self.config.applications_good_30d,
            self.config.applications_excessive_30d,
        )
        offer_risk = 1.0 - min(offer_acceptance / self.config.poor_offer_acceptance_rate, 1.0)

        if applications_risk >= 0.5:
            explanation.append("High recent application volume increases behavioral risk.")
        if response_hours >= self.config.response_time_poor_hours:
            explanation.append("Slow average response time increases engagement risk.")
        if inactive_risk >= 0.5:
            explanation.append("Recent activity is stale or missing.")

        return self._weighted_average(
            {
                "notice_period_days": self._linear_risk(
                    notice_days,
                    self.config.notice_period_low_risk_days,
                    self.config.notice_period_high_risk_days,
                ),
                "avg_response_time_hours": self._linear_risk(
                    response_hours,
                    self.config.response_time_good_hours,
                    self.config.response_time_poor_hours,
                ),
                "applications_submitted_30d": applications_risk,
                "offer_acceptance_rate": offer_risk,
                "inactive_user": inactive_risk,
            },
            self.config.risk_weights,
        )

    def _inactive_risk(self, signals: Mapping[str, Any]) -> float:
        last_active = self._text(signals, "last_active_date")
        if not last_active:
            return 1.0

        active_date = self._parse_date(last_active)
        if active_date is None:
            return 1.0

        days_inactive = (self._today_provider() - active_date).days
        if days_inactive <= self.config.inactive_after_days:
            return 0.0
        return self._linear_risk(days_inactive, self.config.inactive_after_days, self.config.stale_after_days)

    @staticmethod
    def _signals(candidate: Any) -> Mapping[str, Any]:
        if isinstance(candidate, Mapping):
            raw = candidate.get("redrob_signals") or candidate.get("signals") or candidate
            return BehaviorAnalyzer._as_mapping(raw)
        return BehaviorAnalyzer._as_mapping(getattr(candidate, "signals", {}))

    @staticmethod
    def _as_mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        if is_dataclass(value):
            return asdict(value)
        return getattr(value, "__dict__", {})

    @staticmethod
    def _bool(signals: Mapping[str, Any], *keys: str) -> bool:
        for key in keys:
            if key in signals:
                return bool(signals[key])
        return False

    @staticmethod
    def _number(signals: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(signals.get(key, default))
        except (TypeError, ValueError):
            return default

    @classmethod
    def _rate(cls, signals: Mapping[str, Any], key: str) -> float:
        value = cls._number(signals, key)
        if value > 1.0:
            value = value / 100.0
        return cls._clamp(value)

    @classmethod
    def _cap_score(cls, signals: Mapping[str, Any], key: str, cap: float) -> float:
        if cap <= 0:
            return 0.0
        return cls._clamp(cls._number(signals, key) / cap)

    @staticmethod
    def _text(signals: Mapping[str, Any], key: str) -> str:
        value = signals.get(key, "")
        return str(value).strip() if value is not None else ""

    @classmethod
    def _linear_risk(cls, value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return cls._clamp((value - low) / (high - low))

    @classmethod
    def _weighted_average(cls, values: Mapping[str, float], weights: Mapping[str, float]) -> float:
        total_weight = sum(max(weight, 0.0) for weight in weights.values())
        if total_weight <= 0:
            return 0.0
        score = sum(cls._clamp(values.get(name, 0.0)) * max(weight, 0.0) for name, weight in weights.items())
        return cls._clamp(score / total_weight)

    @staticmethod
    @lru_cache(maxsize=4096)
    def _parse_date(value: str) -> date | None:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value[:10], fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None

    @staticmethod
    def _clamp(value: float) -> float:
        return min(max(value, 0.0), 1.0)
