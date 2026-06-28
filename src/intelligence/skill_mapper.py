"""Skill normalization and weighted candidate-to-job skill matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, Iterable, Mapping, Pattern, Sequence, Set

from src.core.models import Candidate, CandidateDNA, JobDNA, SkillMatchResult


@dataclass(frozen=True)
class SkillTaxonomy:
    """Canonical skill vocabulary and aliases used across extraction and scoring."""

    aliases: Dict[str, str] = field(default_factory=lambda: {
        "python": "python",
        "py": "python",
        "llm": "llm",
        "llms": "llm",
        "large language model": "llm",
        "large language models": "llm",
        "lora": "lora",
        "low rank adaptation": "lora",
        "qlora": "qlora",
        "quantized lora": "qlora",
        "peft": "peft",
        "parameter efficient fine tuning": "peft",
        "parameter efficient finetuning": "peft",
        "parameter efficient fine-tuning": "peft",
        "embedding": "embeddings",
        "embeddings": "embeddings",
        "sentence embedding": "embeddings",
        "sentence embeddings": "embeddings",
        "rag": "rag",
        "retrieval augmented generation": "rag",
        "retrieval-augmented generation": "rag",
        "retrieval": "retrieval",
        "retriever": "retrieval",
        "ranking": "ranking",
        "ranker": "ranking",
        "reranking": "ranking",
        "re ranking": "ranking",
        "learning to rank": "ranking",
        "learning-to-rank": "ranking",
        "faiss": "faiss",
        "milvus": "milvus",
        "qdrant": "qdrant",
        "weaviate": "weaviate",
        "pinecone": "pinecone",
        "elasticsearch": "elasticsearch",
        "elastic search": "elasticsearch",
        "opensearch": "opensearch",
        "open search": "opensearch",
        "vector db": "vector_db",
        "vector database": "vector_db",
        "vector databases": "vector_db",
        "vector store": "vector_db",
        "fastapi": "fastapi",
        "fast api": "fastapi",
        "flask": "flask",
        "sql": "sql",
        "spark": "spark",
        "pyspark": "spark",
        "kafka": "kafka",
        "airflow": "airflow",
        "pytorch": "pytorch",
        "torch": "pytorch",
        "tensorflow": "tensorflow",
        "tf": "tensorflow",
        "huggingface": "huggingface",
        "hugging face": "huggingface",
        "hf": "huggingface",
        "transformer": "transformers",
        "transformers": "transformers",
        "sentence transformers": "sentence_transformers",
        "sentence-transformers": "sentence_transformers",
        "sentencetransformers": "sentence_transformers",
        "langchain": "langchain",
        "lang chain": "langchain",
        "deep learning": "deep_learning",
        "machine learning": "machine_learning",
        "ml": "machine_learning",
        "backend": "backend",
    })
    weights: Dict[str, float] = field(default_factory=lambda: {
        "python": 0.88,
        "llm": 0.92,
        "lora": 0.72,
        "qlora": 0.72,
        "peft": 0.70,
        "embeddings": 0.88,
        "rag": 0.90,
        "retrieval": 0.88,
        "ranking": 0.86,
        "vector_db": 0.84,
        "faiss": 0.80,
        "milvus": 0.80,
        "qdrant": 0.78,
        "weaviate": 0.78,
        "pinecone": 0.78,
        "elasticsearch": 0.76,
        "opensearch": 0.76,
        "fastapi": 0.72,
        "flask": 0.66,
        "sql": 0.68,
        "spark": 0.76,
        "kafka": 0.74,
        "airflow": 0.70,
        "pytorch": 0.84,
        "tensorflow": 0.82,
        "huggingface": 0.78,
        "transformers": 0.84,
        "sentence_transformers": 0.82,
        "langchain": 0.76,
        "deep_learning": 0.78,
        "machine_learning": 0.78,
        "backend": 0.70,
    })
    families: Dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "vector_db": ("faiss", "milvus", "qdrant", "weaviate", "pinecone"),
        "retrieval": ("rag", "embeddings", "faiss", "milvus", "qdrant", "weaviate", "pinecone", "elasticsearch", "opensearch"),
        "rag": ("retrieval", "embeddings", "llm"),
        "ranking": ("elasticsearch", "opensearch"),
        "transformers": ("huggingface", "sentence_transformers"),
        "huggingface": ("transformers", "sentence_transformers"),
        "sentence_transformers": ("huggingface", "transformers", "embeddings"),
    })

    def normalize(self, skill: str) -> str:
        key = self._key(skill)
        return self.aliases.get(key, key.replace(" ", "_"))

    def weight_for(self, skill: str) -> float:
        return self.weights.get(self.normalize(skill), 0.50)

    @staticmethod
    @lru_cache(maxsize=8192)
    def _key(skill: str) -> str:
        normalized = skill.lower().strip()
        normalized = re.sub(r"[_\-\/]+", " ", normalized)
        normalized = re.sub(r"[^a-z0-9+#. ]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()


@dataclass(frozen=True)
class CandidateSkillEvidence:
    normalized: Set[str]
    labels: Set[str]
    labels_by_initial: Mapping[str, Set[str]]
    text: str
    term_counts: Mapping[str, int]


class SkillMapper:
    """Scores skill fit using weighted required/preferred matching."""

    REQUIRED_WEIGHT = 0.80
    PREFERRED_WEIGHT = 0.20
    FUZZY_THRESHOLD = 0.88

    def __init__(self, taxonomy: SkillTaxonomy | None = None):
        self.taxonomy = taxonomy or SkillTaxonomy()
        self.aliases = self.taxonomy.aliases
        self._alias_patterns: dict[str, Pattern[str]] = {
            alias: re.compile(self._alias_pattern(alias))
            for alias in self.aliases
        }
        self._alias_keys: dict[str, str] = {
            alias: self.taxonomy._key(alias)
            for alias in self.aliases
        }
        self._alias_prefilters: dict[str, str] = {
            alias: self._alias_keys[alias].split()[0]
            for alias in self.aliases
        }
        self._aliases_by_canonical: dict[str, list[str]] = {}
        for alias, canonical in self.aliases.items():
            self._aliases_by_canonical.setdefault(canonical, []).append(alias)
        self._required_cache: dict[int, Set[str]] = {}
        self._preferred_cache: dict[tuple[int, frozenset[str]], Set[str]] = {}
        self._weight_cache: dict[str, float] = {}

    def evaluate(self, candidate: Candidate, job_dna: JobDNA | CandidateDNA) -> SkillMatchResult:
        evidence = self._candidate_evidence(candidate)
        required = self._required_skills(job_dna)
        preferred = self._preferred_skills(job_dna, required)

        if not required and not preferred:
            return SkillMatchResult(
                score=0.0,
                reasoning=["No required or preferred technical skills were extracted from the JD."],
                explanation=["No required or preferred technical skills were extracted from the JD."],
            )

        matched_required, missing_required = self._partition_matches(required, evidence)
        matched_preferred, missing_preferred = self._partition_matches(preferred, evidence)

        required_score = self._weighted_coverage(matched_required, required)
        preferred_score = self._weighted_coverage(matched_preferred, preferred)
        if not preferred:
            preferred_score = 1.0

        seniority_bonus = self._seniority_bonus(candidate, matched_required | matched_preferred)
        stuffing_penalty, stuffing_reasons = self._keyword_stuffing_penalty(candidate, evidence)

        raw_score = (
            required_score * self.REQUIRED_WEIGHT
            + preferred_score * self.PREFERRED_WEIGHT
            + seniority_bonus
            - stuffing_penalty
        )
        score = round(min(max(raw_score, 0.0), 1.0), 3)

        explanation = [
            f"Required skills contribute 80%; matched {len(matched_required)} of {len(required)}.",
            f"Preferred skills contribute 20%; matched {len(matched_preferred)} of {len(preferred)}.",
        ]
        if seniority_bonus:
            explanation.append("Applied depth bonus from endorsements, assessments, or long skill duration.")
        explanation.extend(stuffing_reasons)

        matched_required_list = sorted(matched_required)
        missing_required_list = sorted(missing_required)
        matched_preferred_list = sorted(matched_preferred)
        missing_preferred_list = sorted(missing_preferred)
        weighted_hits = {
            skill: self.taxonomy.weight_for(skill)
            for skill in sorted(matched_required | matched_preferred)
        }

        return SkillMatchResult(
            score=score,
            matched_skills=matched_required_list,
            missing_skills=missing_required_list,
            matched_required=matched_required_list,
            missing_required=missing_required_list,
            matched_preferred=matched_preferred_list,
            missing_preferred=missing_preferred_list,
            weighted_hits=weighted_hits,
            reasoning=explanation,
            explanation=explanation,
        )

    def match(self, candidate: Candidate, job_dna: JobDNA | CandidateDNA) -> float:
        return self.evaluate(candidate, job_dna).score

    def _candidate_evidence(self, candidate: Candidate) -> CandidateSkillEvidence:
        skill_labels = {skill.name for skill in candidate.skills}
        assessment_labels = set(candidate.signals.skill_assessment_scores.keys())
        text = self._candidate_text(candidate)
        term_counts = self._term_counts(text)
        career_labels = {term for term, count in term_counts.items() if count > 0}
        labels = {self.taxonomy._key(label) for label in skill_labels | assessment_labels if label}
        labels_by_initial: dict[str, Set[str]] = {}
        for label in labels:
            if label:
                labels_by_initial.setdefault(label[0], set()).add(label)
        normalized = {self.taxonomy.normalize(label) for label in labels}
        normalized.update(career_labels)
        normalized.update(term for term, count in term_counts.items() if count > 0)
        return CandidateSkillEvidence(
            normalized=normalized,
            labels=labels,
            labels_by_initial=labels_by_initial,
            text=text,
            term_counts=term_counts,
        )

    def _candidate_text(self, candidate: Candidate) -> str:
        return self.taxonomy._key(" ".join(
            [
                candidate.profile.headline,
                candidate.profile.summary,
                candidate.profile.current_title,
                candidate.profile.current_industry,
            ]
            + [skill.name for skill in candidate.skills]
            + list(candidate.signals.skill_assessment_scores.keys())
            + [job.title for job in candidate.career_history]
            + [job.industry for job in candidate.career_history]
            + [job.description for job in candidate.career_history]
        ))

    def _career_terms(self, text: str) -> Set[str]:
        return {
            canonical
            for alias, canonical in self.taxonomy.aliases.items()
            if self._contains_alias(text, alias)
        }

    def _required_skills(self, job_dna: JobDNA | CandidateDNA) -> Set[str]:
        cache_key = id(job_dna)
        cached = self._required_cache.get(cache_key)
        if cached is not None:
            return cached
        if isinstance(job_dna, JobDNA):
            skills = set(job_dna.required_skills) | set(job_dna.required_tools)
            skills |= {key for key, value in job_dna.technical.items() if value > 0 and key not in set(job_dna.preferred_skills)}
        else:
            skills = {key for key, value in job_dna.technical.items() if value > 0}
        normalized = {self.taxonomy.normalize(skill) for skill in skills}
        self._required_cache[cache_key] = normalized
        return normalized

    def _preferred_skills(self, job_dna: JobDNA | CandidateDNA, required: Set[str]) -> Set[str]:
        cache_key = (id(job_dna), frozenset(required))
        cached = self._preferred_cache.get(cache_key)
        if cached is not None:
            return cached
        if not isinstance(job_dna, JobDNA):
            return set()
        preferred = set(job_dna.preferred_skills) | set(job_dna.preferred_tools)
        normalized = {self.taxonomy.normalize(skill) for skill in preferred} - required
        self._preferred_cache[cache_key] = normalized
        return normalized

    def _partition_matches(
        self,
        target_skills: Iterable[str],
        evidence: CandidateSkillEvidence,
    ) -> tuple[Set[str], Set[str]]:
        matched = {
            skill
            for skill in target_skills
            if self._matches_skill(skill, evidence)
        }
        target_set = set(target_skills)
        return matched, target_set - matched

    def _matches_skill(self, skill: str, evidence: CandidateSkillEvidence) -> bool:
        normalized = self.taxonomy.normalize(skill)
        if normalized in evidence.normalized:
            return True
        family = set(self.taxonomy.families.get(normalized, ()))
        if family & evidence.normalized:
            return True
        aliases = self._aliases_by_canonical.get(normalized, [])
        if any(self._contains_alias(evidence.text, alias) for alias in aliases):
            return True
        if len(normalized) < 5:
            return False
        fuzzy_labels = evidence.labels_by_initial.get(normalized.replace("_", " ")[:1], set())
        return any(
            self._fuzzy_equal(normalized, label) or self._fuzzy_equal(normalized.replace("_", " "), label)
            for label in fuzzy_labels
        )

    def _weighted_coverage(self, matched: Set[str], total: Set[str]) -> float:
        total_weight = sum(self._weight_for(skill) for skill in total)
        if total_weight <= 0:
            return 0.0
        hit_weight = sum(self._weight_for(skill) for skill in matched)
        return hit_weight / total_weight

    def _seniority_bonus(self, candidate: Candidate, matched: Iterable[str]) -> float:
        matched_set = {self.taxonomy.normalize(skill) for skill in matched}
        if not matched_set:
            return 0.0
        duration_bonus = 0.0
        endorsement_bonus = 0.0
        assessment_bonus = 0.0
        for skill in candidate.skills:
            normalized = self.taxonomy.normalize(skill.name)
            if normalized in matched_set:
                duration_bonus += min(skill.duration_months / 1200, 0.015)
                endorsement_bonus += min(skill.endorsements / 1000, 0.010)
        for skill, score in candidate.signals.skill_assessment_scores.items():
            if self.taxonomy.normalize(skill) in matched_set and score >= 80:
                assessment_bonus += 0.01
        return min(duration_bonus + endorsement_bonus + assessment_bonus, 0.08)

    def _keyword_stuffing_penalty(
        self,
        candidate: Candidate,
        evidence: CandidateSkillEvidence,
    ) -> tuple[float, list[str]]:
        reasons = []
        repeated_terms = [term for term, count in evidence.term_counts.items() if count >= 5]
        known_mentions = sum(evidence.term_counts.values())
        unique_known = len([term for term, count in evidence.term_counts.items() if count > 0])
        explicit_skill_count = len(candidate.skills)
        years = candidate.profile.years_of_experience

        penalty = 0.0
        if repeated_terms:
            penalty += min(0.03 * len(repeated_terms), 0.12)
            reasons.append("Applied keyword-stuffing penalty for repeated skill mentions: " + ", ".join(sorted(repeated_terms)[:6]) + ".")
        if unique_known >= 18 and years < 3:
            penalty += 0.08
            reasons.append("Applied keyword-stuffing penalty for unusually broad AI skill coverage relative to experience.")
        if explicit_skill_count >= 24 and years < 5:
            penalty += 0.06
            reasons.append("Applied keyword-stuffing penalty for an unusually long skill list relative to experience.")
        if known_mentions >= 45 and unique_known <= 10:
            penalty += 0.06
            reasons.append("Applied keyword-stuffing penalty for dense repeated keyword usage.")
        return min(penalty, 0.25), reasons

    def _term_counts(self, text: str) -> Dict[str, int]:
        counts = {canonical: 0 for canonical in self.taxonomy.weights}
        padded_text = f" {text} "
        for alias, canonical in self.taxonomy.aliases.items():
            alias_key = self._alias_keys[alias]
            if self._alias_prefilters[alias] not in text:
                continue
            count = padded_text.count(f" {alias_key} ")
            if count:
                counts[canonical] = counts.get(canonical, 0) + count
        return counts

    def _contains_alias(self, text: str, alias: str) -> bool:
        alias_key = self._alias_keys[alias]
        if self._alias_prefilters[alias] not in text:
            return False
        return f" {alias_key} " in f" {text} "

    @staticmethod
    def _alias_pattern(alias: str) -> str:
        escaped = re.escape(SkillTaxonomy._key(alias))
        escaped = escaped.replace(r"\ ", r"[\s\-_\/]+")
        return rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"

    @staticmethod
    def _fuzzy_equal(target: str, label: str) -> bool:
        clean_target = target.replace("_", " ")
        clean_label = label.replace("_", " ")
        if abs(len(clean_target) - len(clean_label)) > max(4, len(clean_target) // 2):
            return False
        matcher = SequenceMatcher(None, clean_target, clean_label)
        if matcher.real_quick_ratio() < SkillMapper.FUZZY_THRESHOLD:
            return False
        if matcher.quick_ratio() < SkillMapper.FUZZY_THRESHOLD:
            return False
        return matcher.ratio() >= SkillMapper.FUZZY_THRESHOLD

    def _weight_for(self, skill: str) -> float:
        cached = self._weight_cache.get(skill)
        if cached is None:
            cached = self.taxonomy.weight_for(skill)
            self._weight_cache[skill] = cached
        return cached
