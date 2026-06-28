"""Lightweight wall-clock profiling for CLI and pipeline stages."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ProfileStage:
    """Aggregated timing for one named stage."""

    name: str
    seconds: float
    calls: int


class StageProfiler:
    """Accumulates elapsed time by stage without changing ranking behavior."""

    def __init__(self) -> None:
        self._seconds: defaultdict[str, float] = defaultdict(float)
        self._calls: defaultdict[str, int] = defaultdict(int)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started_at)

    def add(self, name: str, seconds: float) -> None:
        self._seconds[name] += seconds
        self._calls[name] += 1

    def register(self, name: str) -> None:
        self._seconds[name] += 0.0
        self._calls[name] += 0

    def stages(self) -> list[ProfileStage]:
        return [
            ProfileStage(name=name, seconds=seconds, calls=self._calls[name])
            for name, seconds in self._seconds.items()
        ]

    def sorted_stages(self) -> list[ProfileStage]:
        return sorted(self.stages(), key=lambda stage: stage.seconds, reverse=True)

    def print_summary(self) -> None:
        stages = self.sorted_stages()
        print("=" * 70)
        print("PROFILE SUMMARY")
        print("=" * 70)
        if not stages:
            print("No profiling data collected.")
            return
        total = sum(stage.seconds for stage in stages)
        print(f"{'Stage':<34} {'Time':>10} {'Percentage':>12}")
        print("-" * 70)
        for stage in stages:
            share = 0.0 if total <= 0 else (stage.seconds / total) * 100
            print(f"{stage.name:<34} {stage.seconds:>9.3f}s {share:>10.1f}%")
