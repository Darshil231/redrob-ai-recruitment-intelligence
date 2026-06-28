"""Fast deterministic eligibility filtering before costly ranking stages."""

from src.core.models import Candidate, EligibilityResult, JobDNA, SkillMatchResult
from src.intelligence.skill_mapper import SkillMapper


class EligibilityFilter:
    """Applies hard and soft recruiter filters for scale-first screening."""

    def __init__(self, skill_mapper: SkillMapper | None = None):
        self.skill_mapper = skill_mapper or SkillMapper()

    def evaluate(
        self,
        candidate: Candidate,
        job_dna: JobDNA,
        skill_result: SkillMatchResult | None = None,
    ) -> EligibilityResult:
        reasons: list[str] = []
        rejections: list[str] = []
        checks = [
            self._experience(candidate, job_dna, reasons, rejections),
            self._title(candidate, job_dna, reasons, rejections),
            self._notice_period(candidate, job_dna, reasons, rejections),
            self._location(candidate, job_dna, reasons, rejections),
            self._skills(candidate, job_dna, reasons, rejections, skill_result),
            self._education(candidate, job_dna, reasons, rejections),
        ]
        score = round(sum(checks) / len(checks), 3)
        hard_fail = any(reason.startswith("Hard fail") for reason in rejections)
        return EligibilityResult(
            passed=not hard_fail and score >= 0.45,
            score=score,
            reasons=reasons,
            rejection_reasons=rejections,
        )

    def _experience(self, candidate: Candidate, job_dna: JobDNA, reasons: list[str], rejections: list[str]) -> float:
        years = candidate.profile.years_of_experience
        if years < job_dna.experience_min:
            rejections.append(f"Hard fail: {years:.1f} years below required {job_dna.experience_min}.")
            return 0.0
        if job_dna.experience_max and years > job_dna.experience_max + 6:
            rejections.append(f"Potential overlevel: {years:.1f} years exceeds target range.")
            return 0.65
        reasons.append("Experience fits the JD range.")
        return 1.0

    def _title(self, candidate: Candidate, job_dna: JobDNA, reasons: list[str], rejections: list[str]) -> float:
        if not job_dna.title_keywords:
            return 1.0
        title = candidate.profile.current_title.lower()
        career_titles = " ".join(job.title.lower() for job in candidate.career_history)
        if any(keyword in title or keyword in career_titles for keyword in job_dna.title_keywords):
            reasons.append("Current or past title aligns with the role.")
            return 1.0
        rejections.append("Title history does not clearly align with the role.")
        return 0.35

    def _notice_period(self, candidate: Candidate, job_dna: JobDNA, reasons: list[str], rejections: list[str]) -> float:
        if candidate.signals.notice_period_days <= job_dna.notice_period_max_days:
            reasons.append("Notice period is within target.")
            return 1.0
        rejections.append("Notice period exceeds target.")
        return 0.45

    def _location(self, candidate: Candidate, job_dna: JobDNA, reasons: list[str], rejections: list[str]) -> float:
        if not job_dna.location or job_dna.work_mode == "remote":
            return 1.0
        candidate_location = f"{candidate.profile.location} {candidate.profile.country}".lower()
        target = job_dna.location.lower()
        if target in candidate_location or candidate.signals.willing_to_relocate:
            reasons.append("Location is compatible or candidate can relocate.")
            return 1.0
        rejections.append("Location does not match and relocation is not indicated.")
        return 0.35

    def _skills(
        self,
        candidate: Candidate,
        job_dna: JobDNA,
        reasons: list[str],
        rejections: list[str],
        skill_result: SkillMatchResult | None = None,
    ) -> float:
        skill_result = skill_result or self.skill_mapper.evaluate(candidate, job_dna)
        if skill_result.score >= 0.35:
            reasons.append("Required skill coverage is strong enough for ranking.")
            return skill_result.score
        rejections.append("Skill coverage is below preferred threshold.")
        return skill_result.score

    def _education(self, candidate: Candidate, job_dna: JobDNA, reasons: list[str], rejections: list[str]) -> float:
        if not job_dna.required_education:
            return 1.0
        education_text = " ".join(
            f"{item.degree} {item.field_of_study} {item.institution}".lower()
            for item in candidate.education
        )
        if any(degree in education_text for degree in job_dna.required_education):
            reasons.append("Education requirement is met.")
            return 1.0
        rejections.append("Required education was not found.")
        return 0.45
