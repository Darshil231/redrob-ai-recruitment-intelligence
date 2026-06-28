"""Application service for ranking jobs and retaining latest results for APIs."""

import os
import time
import tracemalloc

from src.core.config import AppConfig
from src.core.models import Candidate, RankedCandidate
from src.ranking.pipeline import RankingPipeline
from src.utils.profiler import StageProfiler


class RankingService:
    """Thin use-case service over the ranking pipeline."""

    def __init__(
        self,
        pipeline: RankingPipeline | None = None,
        config: AppConfig | None = None,
        profiler: StageProfiler | None = None,
    ):
        self.config = config or AppConfig()
        self.profiler = profiler
        self.pipeline = pipeline or RankingPipeline(self.config, profiler=profiler)
        self._latest_results: list[RankedCandidate] = []
        self._latest_metrics: dict[str, object] = {}

    def rank(self, jd_text: str, candidates: list[Candidate]) -> list[RankedCandidate]:
        started_at = time.perf_counter()
        trace_memory = os.getenv("REDROB_TRACE_MEMORY") == "1"
        peak_bytes = 0
        if trace_memory:
            tracemalloc.start()
        try:
            self._latest_results = self.pipeline.rank(jd_text, candidates)
            if trace_memory:
                _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            if trace_memory:
                tracemalloc.stop()
        execution_time_seconds = time.perf_counter() - started_at
        self._latest_metrics = {
            **self.pipeline.latest_metrics,
            "execution_time_seconds": round(execution_time_seconds, 3),
            "memory_usage_mb": round(peak_bytes / (1024 * 1024), 3),
        }
        return self._latest_results

    def latest(self) -> list[RankedCandidate]:
        return self._latest_results

    def latest_metrics(self) -> dict[str, object]:
        return self._latest_metrics
