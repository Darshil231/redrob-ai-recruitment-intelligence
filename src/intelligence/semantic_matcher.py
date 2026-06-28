"""Semantic ranking using SentenceTransformer embeddings over shortlisted candidates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from src.core.config import SemanticConfig
from src.core.models import Candidate

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SemanticMatcher:
    """Batched semantic scorer with lazy model loading and disk embedding cache."""

    _MODEL_CACHE: dict[tuple[str, str | None], "SentenceTransformer"] = {}

    def __init__(self, config: SemanticConfig | None = None):
        self.config = config or SemanticConfig()
        self._model: SentenceTransformer | None = None
        self._jd_cache: dict[str, np.ndarray] = {}
        self._candidate_text_cache: dict[int, str] = {}
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self._device()
            cache_key = (self.config.model_name, device)
            if cache_key not in self._MODEL_CACHE:
                self._MODEL_CACHE[cache_key] = SentenceTransformer(
                    self.config.model_name,
                    device=device,
                    local_files_only=self.config.local_files_only,
                )
            self._model = self._MODEL_CACHE[cache_key]
        return self._model

    def build_candidate_text(self, candidate: Candidate) -> str:
        cache_key = id(candidate)
        if cache_key in self._candidate_text_cache:
            return self._candidate_text_cache[cache_key]

        parts = [
            candidate.profile.headline,
            candidate.profile.summary,
            candidate.profile.current_title,
            candidate.profile.current_industry,
            " ".join(skill.name for skill in candidate.skills),
        ]
        parts.extend(f"{job.title} {job.industry} {job.description}" for job in candidate.career_history)
        parts.extend(f"{edu.degree} {edu.field_of_study}" for edu in candidate.education)
        text = "\n".join(part for part in parts if part)
        self._candidate_text_cache[cache_key] = text
        return text

    def similarity(self, candidate: Candidate, jd_text: str) -> float:
        return float(self.batch_similarity([candidate], jd_text)[0])

    def batch_similarity(self, candidates: Sequence[Candidate], jd_text: str) -> np.ndarray:
        if not candidates:
            return np.array([], dtype=np.float32)

        jd_embedding = self._embed_jd(jd_text)
        candidate_embeddings = self._embed_candidates(candidates)
        scores = candidate_embeddings @ jd_embedding
        return np.clip(scores, 0.0, 1.0)

    def _embed_jd(self, jd_text: str) -> np.ndarray:
        key = self._hash_text(jd_text)
        if key not in self._jd_cache:
            self._jd_cache[key] = self._encode([jd_text])[0]
        return self._jd_cache[key]

    def _embed_candidates(self, candidates: Sequence[Candidate]) -> np.ndarray:
        embeddings: list[np.ndarray | None] = []
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, candidate in enumerate(candidates):
            candidate_text = self.build_candidate_text(candidate)
            cache_path = self._candidate_cache_path(candidate, candidate_text)
            if cache_path.exists():
                embeddings.append(np.load(cache_path))
            else:
                embeddings.append(None)
                missing_indices.append(index)
                missing_texts.append(candidate_text)

        if missing_texts:
            encoded = self._encode(missing_texts)
            for index, candidate_text, embedding in zip(missing_indices, missing_texts, encoded):
                embeddings[index] = embedding
                np.save(self._candidate_cache_path(candidates[index], candidate_text), embedding)

        return np.vstack([embedding for embedding in embeddings if embedding is not None])

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def _candidate_cache_path(self, candidate: Candidate, candidate_text: str | None = None) -> Path:
        text_hash = self._hash_text(candidate_text if candidate_text is not None else self.build_candidate_text(candidate))
        return self.config.cache_dir / f"{candidate.candidate_id}-{text_hash}.npy"

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _device() -> str | None:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return None
