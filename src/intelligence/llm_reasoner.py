"""Deterministic recruiter-style reasoning for top candidates."""

from __future__ import annotations

from src.core.models import (
    Candidate,
    CareerAnalysisResult,
    JobDNA,
    LLMReasoningResult,
    SkillMatchResult,
)


class LLMReasoner:
    """Produces concise recruiter reasoning without external API calls."""

    MAX_WORDS = 150

    def reason(
        self,
        candidate: Candidate,
        job_dna: JobDNA,
        semantic_score: float,
        skill_match: SkillMatchResult,
        career: CareerAnalysisResult,
        filter_score: float,
    ) -> LLMReasoningResult:
        confidence = self._confidence(skill_match.score, career.score, semantic_score, filter_score)
        matched_skills = self._top_items(skill_match.matched_skills, 8)
        missing_skills = self._top_items(skill_match.missing_skills, 6)
        relevant_experience = self._top_items(career.relevant_experience, 3)
        behavior_summary = self._behavior_summary(candidate)
        strengths = self._strengths(matched_skills, relevant_experience, semantic_score, candidate)
        weaknesses = self._weaknesses(missing_skills, career, filter_score, candidate)
        recommendation = self._recommendation(confidence, weaknesses)
        risk_assessment = self._risk_assessment(missing_skills, career, filter_score)
        summary = self._summary(
            candidate=candidate,
            job_dna=job_dna,
            confidence=confidence,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            relevant_experience=relevant_experience,
            behavior_summary=behavior_summary,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendation=recommendation,
        )

        return LLMReasoningResult(
            overall_match_percent=round(confidence * 100, 1),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            relevant_experience=relevant_experience,
            behavior_summary=behavior_summary,
            strengths=strengths,
            weaknesses=weaknesses,
            hiring_recommendation=recommendation,
            risk_assessment=risk_assessment,
            summary=summary,
            confidence=confidence,
        )

    @staticmethod
    def _confidence(skill_score: float, career_score: float, semantic_score: float, filter_score: float) -> float:
        return round(
            min(
                skill_score * 0.35
                + career_score * 0.25
                + semantic_score * 0.25
                + filter_score * 0.15,
                1.0,
            ),
            3,
        )

    def _summary(
        self,
        *,
        candidate: Candidate,
        job_dna: JobDNA,
        confidence: float,
        matched_skills: list[str],
        missing_skills: list[str],
        relevant_experience: list[str],
        behavior_summary: str,
        strengths: list[str],
        weaknesses: list[str],
        recommendation: str,
    ) -> str:
        sections = [
            f"Overall Match: {round(confidence * 100, 1)}% for {job_dna.role}.",
            "Matched Skills: " + self._phrase_list(matched_skills, "No direct required-skill match found.") ,
            "Missing Skills: " + self._phrase_list(missing_skills, "No major required gaps found."),
            "Relevant Experience: " + self._phrase_list(relevant_experience, "Limited role-specific career evidence."),
            "Behavior Summary: " + behavior_summary,
            "Strengths: " + self._phrase_list(strengths, "Some baseline alignment exists."),
            "Weaknesses: " + self._phrase_list(weaknesses, "No material weakness flagged."),
            "Hiring Recommendation: " + recommendation,
            f"Confidence: {confidence:.2f}.",
        ]
        intro = (
            f"{candidate.profile.anonymized_name or candidate.candidate_id} is a "
            f"{candidate.profile.current_title or 'candidate'} with "
            f"{candidate.profile.years_of_experience:.1f} years of experience."
        )
        return self._limit_words(" ".join([intro] + sections), self.MAX_WORDS)

    @staticmethod
    def _strengths(
        matched_skills: list[str],
        relevant_experience: list[str],
        semantic_score: float,
        candidate: Candidate,
    ) -> list[str]:
        strengths = []
        if matched_skills:
            strengths.append("Strong required-skill coverage: " + ", ".join(matched_skills[:5]) + ".")
        if relevant_experience:
            strengths.append("Relevant career evidence is present.")
        if semantic_score >= 0.65:
            strengths.append("Profile language is close to the job description.")
        if candidate.signals.open_to_work:
            strengths.append("Candidate is open to work.")
        return strengths or ["Partial fit with limited explicit evidence."]

    @staticmethod
    def _weaknesses(
        missing_skills: list[str],
        career: CareerAnalysisResult,
        filter_score: float,
        candidate: Candidate,
    ) -> list[str]:
        weaknesses = []
        if missing_skills:
            weaknesses.append("Missing explicit skills: " + ", ".join(missing_skills[:5]) + ".")
        if career.score < 0.45:
            weaknesses.append("Career history has limited direct relevance.")
        if filter_score < 0.70:
            weaknesses.append("Eligibility fit needs validation.")
        if candidate.signals.notice_period_days > 60:
            weaknesses.append("Notice period may slow hiring.")
        return weaknesses

    @staticmethod
    def _recommendation(confidence: float, weaknesses: list[str]) -> str:
        if confidence >= 0.78 and len(weaknesses) <= 1:
            return "Prioritize recruiter outreach."
        if confidence >= 0.62:
            return "Worth recruiter screen; validate gaps on the first call."
        if confidence >= 0.48:
            return "Keep as backup unless the pipeline is thin."
        return "Do not prioritize for this role."

    @staticmethod
    def _risk_assessment(
        missing_skills: list[str],
        career: CareerAnalysisResult,
        filter_score: float,
    ) -> str:
        risks = []
        if len(missing_skills) >= 4:
            risks.append("skill coverage")
        if career.score < 0.40:
            risks.append("career relevance")
        if filter_score < 0.60:
            risks.append("eligibility fit")
        return "Low risk." if not risks else "Risk areas: " + ", ".join(risks) + "."

    @staticmethod
    def _behavior_summary(candidate: Candidate) -> str:
        signals = candidate.signals
        availability = "open to work" if signals.open_to_work else "not marked open to work"
        notice = f"{signals.notice_period_days}-day notice"
        response = "strong recruiter response" if signals.recruiter_response_rate >= 0.70 else "modest recruiter response"
        trust = "verified contact details" if signals.verified_email and signals.verified_phone else "verification needs review"
        return f"{availability}, {notice}, {response}, {trust}."

    @staticmethod
    def _top_items(items: list[str], limit: int) -> list[str]:
        return [item for item in items if item][:limit]

    @staticmethod
    def _phrase_list(items: list[str], fallback: str) -> str:
        return ", ".join(items) + "." if items else fallback

    @staticmethod
    def _limit_words(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        trimmed = " ".join(words[:max_words]).rstrip(" ,;:")
        return trimmed.rstrip(".") + "."
