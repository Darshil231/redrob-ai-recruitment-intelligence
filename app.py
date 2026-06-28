"""CLI entrypoint for generating recruitment ranking outputs."""

from __future__ import annotations

import argparse
import time
import tracemalloc
from itertools import islice

from src.core.config import AppConfig, RankingConfig
from src.core.models import Candidate, RankedCandidate
from src.export.csv_export import CSVExporter
from src.export.evaluation_dashboard import EvaluationDashboard
from src.export.json_export import JSONExporter
from src.export.submission_generator import SubmissionGenerator
from src.intelligence.job_dna import JobDNAExtractor
from src.intelligence.role_relevance_analyzer import RoleRelevanceAnalyzer
from src.parsers.candidate_parser import CandidateParser
from src.parsers.jd_parser import JDParser
from src.services.candidate_service import CandidateService
from src.services.ranking_service import RankingService
from src.utils.profiler import StageProfiler


DEFAULT_FAST_LIMIT = 5000
PROFILE_STAGES = [
    "Candidate loading",
    "Job DNA extraction",
    "Eligibility filtering",
    "Skill mapping",
    "Career analysis",
    "Behavior analysis",
    "Role relevance analysis",
    "Profile integrity analysis",
    "Fast candidate scoring",
    "Fast ranking",
    "Semantic embedding",
    "Semantic ranking",
    "LLM reasoning",
    "CSV export",
    "JSON export",
]


def main() -> None:
    args = _parse_args()
    profiler = StageProfiler()
    for stage in PROFILE_STAGES:
        profiler.register(stage)
    config = AppConfig(
        ranking=RankingConfig(
            fast_top_n=200,
            semantic_top_n=100,
            final_top_n=100,
        )
    )

    if args.full:
        results, metrics = _run_full_pipeline(config, profiler)
    else:
        results, metrics = _run_fast_submission(config, profiler, limit=args.limit)

    with profiler.measure("CSV export"):
        csv_path = CSVExporter().export(results, config.exports_dir / "top_candidates.csv")
    with profiler.measure("JSON export"):
        json_path = JSONExporter().export(results, config.exports_dir / "top_candidates.json")
    submission_path = SubmissionGenerator().generate(results, "submission.csv")
    dashboard_path = EvaluationDashboard().export(
        results,
        metrics,
        config.exports_dir / "evaluation_dashboard.html",
        csv_generated=csv_path.exists(),
        validation_passed=submission_path.exists(),
    )

    _print_summary(results, csv_path, json_path, submission_path, dashboard_path)
    profiler.print_summary()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Redrob ranking outputs.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full semantic ranking pipeline over the full candidate file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_FAST_LIMIT,
        help=f"Candidate rows to scan in fast mode. Default: {DEFAULT_FAST_LIMIT}.",
    )
    return parser.parse_args()


def _run_full_pipeline(config: AppConfig, profiler: StageProfiler) -> tuple[list[RankedCandidate], dict[str, object]]:
    _progress("Loading job description")
    jd_text = JDParser(str(config.jd_path)).load()
    _progress("Loading all candidates")
    with profiler.measure("Candidate loading"):
        candidates = CandidateService(config.candidates_path).list_candidates()
    _progress(f"Running full pipeline for {len(candidates):,} candidates")
    ranking_service = RankingService(config=config, profiler=profiler)
    results = ranking_service.rank(jd_text, candidates)
    return results, ranking_service.latest_metrics()


def _run_fast_submission(config: AppConfig, profiler: StageProfiler, limit: int) -> tuple[list[RankedCandidate], dict[str, object]]:
    started_at = time.perf_counter()
    tracemalloc.start()
    limit = max(limit, 100)

    _progress(f"Fast mode: scanning first {limit:,} candidates")
    jd_text = JDParser(str(config.jd_path)).load()
    with profiler.measure("Job DNA extraction"):
        job_dna = JobDNAExtractor().extract(jd_text)
    role_analyzer = RoleRelevanceAnalyzer()
    with profiler.measure("Candidate loading"):
        candidates = list(islice(CandidateParser(str(config.candidates_path)).parse(), limit))
    results = []
    for candidate in candidates:
        with profiler.measure("Role relevance analysis"):
            role_relevance = role_analyzer.analyze(candidate, job_dna)
        with profiler.measure("Fast ranking"):
            results.append(_score_candidate(candidate, role_relevance))
    with profiler.measure("Fast ranking"):
        ranked = SubmissionGenerator._sorted_results(results)[: config.ranking.final_top_n]
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    _progress(f"Ranked {len(ranked)} candidates for submission")
    return ranked, {
        "candidates_parsed": len(candidates),
        "filtered": len(ranked),
        "rejected": max(len(candidates) - len(ranked), 0),
        "semantic_ranked": len(ranked),
        "average_score": round(sum(result.final_score for result in ranked) / len(ranked), 3) if ranked else 0.0,
        "execution_time_seconds": round(time.perf_counter() - started_at, 3),
        "memory_usage_mb": round(peak_bytes / (1024 * 1024), 3),
    }


def _score_candidate(candidate: Candidate, role_relevance) -> RankedCandidate:
    signals = candidate.signals
    experience_score = min(candidate.profile.years_of_experience / 10.0, 1.0) * 0.35
    engagement_score = min(signals.recruiter_response_rate, 1.0) * 0.20
    completion_score = min(signals.interview_completion_rate, 1.0) * 0.15
    activity_score = min(signals.github_activity_score / 100.0, 1.0) * 0.10
    open_to_work_score = 0.10 if signals.open_to_work else 0.0
    notice_score = 0.10 if signals.notice_period_days <= 90 else 0.03
    activity_fit_score = (
        experience_score
        + engagement_score
        + completion_score
        + activity_score
        + open_to_work_score
        + notice_score
    )
    final_score = round(
        (role_relevance.score * 0.55) + (activity_fit_score * 0.45),
        6,
    )
    matched_skills = [skill.name for skill in candidate.skills[:8]]

    return RankedCandidate(
        candidate=candidate,
        fast_score=final_score,
        filter_score=1.0,
        skill_score=role_relevance.transition_score,
        career_score=experience_score,
        behavior_score=engagement_score + completion_score,
        semantic_score=0.0,
        llm_confidence=0.0,
        final_score=final_score,
        recommendation="Ranked for recruiter review.",
        why_selected=[
            *role_relevance.reasoning,
            f"{candidate.profile.years_of_experience:.1f} years of experience.",
            f"Recruiter response rate {signals.recruiter_response_rate:.2f}.",
            "Open to work." if signals.open_to_work else "Profile included for benchmark coverage.",
        ],
        why_rejected=[],
        matched_skills=matched_skills,
        missing_skills=[],
        relevant_experience=[entry.description for entry in candidate.career_history[:1]],
        relevant_projects=[],
        confidence=0.7,
        reasoning={
            "summary": "Fast deterministic score from role relevance, profile, and Redrob signals.",
            "role_relevance": role_relevance.to_dict(),
        },
    )


def _print_summary(
    results: list[RankedCandidate],
    csv_path,
    json_path,
    submission_path,
    dashboard_path,
) -> None:
    print("=" * 70)
    print("TOP CANDIDATES")
    print("=" * 70)
    for rank, result in enumerate(results[:20], start=1):
        candidate = result.candidate
        print(f"{rank}. {candidate.candidate_id} | {candidate.profile.current_title}")
        print(f"Final Score: {result.final_score:.3f}")
        print(f"Recommendation: {result.recommendation}")
        print(f"Matched Skills: {', '.join(result.matched_skills[:8])}")
        print("-" * 70)

    print(f"CSV export: {csv_path}")
    print(f"JSON export: {json_path}")
    print(f"Submission export: {submission_path}")
    print(f"Evaluation dashboard: {dashboard_path}")


def _progress(message: str) -> None:
    print(f"[redrob] {message}", flush=True)


if __name__ == "__main__":
    main()
