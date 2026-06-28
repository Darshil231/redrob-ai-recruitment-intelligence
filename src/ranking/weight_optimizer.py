"""Grid-search optimizer for ranking component weights."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from math import log2
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.core.config import RankingConfig


@dataclass(frozen=True)
class WeightOptimizationResult:
    """Best weight set and optimization diagnostics."""

    best_weights: dict[str, float]
    best_score: float
    objective: str
    evaluated_candidates: int
    evaluated_weight_sets: int
    config: RankingConfig
    stored_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"] = asdict(self.config)
        return payload


class WeightOptimizer:
    """Optimizes ranking weights with deterministic grid search."""

    COMPONENTS = ("skill", "career", "behavior", "semantic", "reasoning")
    CONFIG_FIELDS = {
        "skill": "skill_weight",
        "career": "career_weight",
        "behavior": "behavior_weight",
        "semantic": "semantic_weight",
        "reasoning": "reasoning_weight",
    }

    def __init__(
        self,
        base_config: RankingConfig | None = None,
        *,
        grid_step: float = 0.10,
        store_path: str | Path = Path("config") / "optimized_ranking_weights.json",
    ):
        if grid_step <= 0 or grid_step > 1:
            raise ValueError("grid_step must be in the range (0, 1].")
        self.base_config = base_config or RankingConfig()
        self.grid_step = grid_step
        self.store_path = Path(store_path)

    def optimize(
        self,
        sample_candidates: Sequence[Any],
        ground_truth: Mapping[str, float] | Sequence[float] | None = None,
        *,
        store: bool = True,
    ) -> WeightOptimizationResult:
        samples = [self._sample(candidate, index) for index, candidate in enumerate(sample_candidates)]
        if not samples:
            raise ValueError("sample_candidates must contain at least one candidate.")

        labels = self._labels(samples, ground_truth)
        objective = "ndcg" if labels else "unsupervised_margin"

        best_weights: dict[str, float] | None = None
        best_score = float("-inf")
        evaluated = 0
        for weights in self._weight_grid():
            evaluated += 1
            predictions = [
                (sample["id"], self._weighted_score(sample["scores"], weights))
                for sample in samples
            ]
            score = self._supervised_score(predictions, labels) if labels else self._unsupervised_margin(samples, weights)
            if score > best_score:
                best_score = score
                best_weights = weights

        if best_weights is None:
            raise ValueError("No valid weight combinations were generated.")

        config = self._config_with(best_weights)
        stored_path = ""
        result = WeightOptimizationResult(
            best_weights=best_weights,
            best_score=round(best_score, self.base_config.score_precision),
            objective=objective,
            evaluated_candidates=len(samples),
            evaluated_weight_sets=evaluated,
            config=config,
        )
        if store:
            stored_path = str(self._store(result))
            result = replace(result, stored_path=stored_path)
        return result

    def _weight_grid(self) -> Iterable[dict[str, float]]:
        ticks = [round(index * self.grid_step, 10) for index in range(int(1 / self.grid_step) + 1)]
        seen = set()
        for values in product(ticks, repeat=len(self.COMPONENTS)):
            total = sum(values)
            if total <= 0:
                continue
            weights = tuple(round(value / total, self.base_config.score_precision) for value in values)
            if weights in seen:
                continue
            seen.add(weights)
            yield dict(zip(self.COMPONENTS, weights))

    def _config_with(self, weights: Mapping[str, float]) -> RankingConfig:
        updates = {
            field_name: weights[component]
            for component, field_name in self.CONFIG_FIELDS.items()
        }
        return replace(self.base_config, **updates)

    def _store(self, result: WeightOptimizationResult) -> Path:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return self.store_path

    def _labels(
        self,
        samples: Sequence[dict[str, Any]],
        ground_truth: Mapping[str, float] | Sequence[float] | None,
    ) -> dict[str, float]:
        if ground_truth is None:
            return {}
        if isinstance(ground_truth, Mapping):
            return {
                sample["id"]: float(ground_truth[sample["id"]])
                for sample in samples
                if sample["id"] in ground_truth
            }
        return {
            sample["id"]: float(label)
            for sample, label in zip(samples, ground_truth)
        }

    def _sample(self, candidate: Any, index: int) -> dict[str, Any]:
        candidate_id = self._value(candidate, "candidate_id", default=f"sample-{index}")
        if candidate_id == f"sample-{index}":
            nested = self._value(candidate, "candidate", default=None)
            candidate_id = self._value(nested, "candidate_id", default=candidate_id)
        return {
            "id": str(candidate_id),
            "scores": {
                component: self._component_score(candidate, component)
                for component in self.COMPONENTS
            },
        }

    def _component_score(self, candidate: Any, component: str) -> float:
        if component == "reasoning":
            value = self._value(candidate, "reasoning_score", default=None)
            if value is None:
                value = self._value(candidate, "llm_confidence", default=self.base_config.legacy_reasoning_confidence)
        else:
            value = self._value(candidate, f"{component}_score", default=0.0)
        return self._clamp(float(value))

    @staticmethod
    def _value(source: Any, key: str, default: Any = None) -> Any:
        if source is None:
            return default
        if isinstance(source, Mapping):
            return source.get(key, default)
        return getattr(source, key, default)

    def _weighted_score(self, scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
        return sum(self._clamp(scores.get(component, 0.0)) * weights[component] for component in self.COMPONENTS)

    def _ndcg(self, predictions: Sequence[tuple[str, float]], labels: Mapping[str, float]) -> float:
        scored = [
            (candidate_id, score, labels.get(candidate_id, 0.0))
            for candidate_id, score in predictions
        ]
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        ideal = sorted(scored, key=lambda item: item[2], reverse=True)
        ideal_dcg = self._dcg([relevance for _, _, relevance in ideal])
        if ideal_dcg <= 0:
            return 0.0
        return self._dcg([relevance for _, _, relevance in ranked]) / ideal_dcg

    def _supervised_score(self, predictions: Sequence[tuple[str, float]], labels: Mapping[str, float]) -> float:
        ndcg = self._ndcg(predictions, labels)
        max_label = max(labels.values(), default=0.0)
        if max_label <= 0:
            return ndcg
        mse = sum(
            (score - self._clamp(labels.get(candidate_id, 0.0) / max_label)) ** 2
            for candidate_id, score in predictions
        ) / max(len(predictions), 1)
        return ndcg - (mse * 0.01)

    @staticmethod
    def _dcg(relevances: Sequence[float]) -> float:
        return sum(relevance / log2(rank + 2) for rank, relevance in enumerate(relevances))

    def _unsupervised_margin(self, samples: Sequence[dict[str, Any]], weights: Mapping[str, float]) -> float:
        scores = sorted(self._weighted_score(sample["scores"], weights) for sample in samples)
        if len(scores) == 1:
            return scores[0]
        spread = scores[-1] - scores[0]
        mean_score = sum(scores) / len(scores)
        return (spread * 0.70) + (mean_score * 0.30)

    def _clamp(self, value: float) -> float:
        return min(max(value, self.base_config.score_floor), self.base_config.score_ceiling)
