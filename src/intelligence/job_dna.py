"""Extracts structured role requirements from a job description."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from src.core.models import JobDNA
from src.intelligence.skill_mapper import SkillTaxonomy


@dataclass(frozen=True)
class SignalSpec:
    """Declarative extraction target with lexical aliases and semantic cues."""

    label: str
    aliases: tuple[str, ...]
    cues: tuple[str, ...] = ()
    weight: float = 0.70


class JobDNAExtractor:
    """Production-oriented JD intelligence engine with explainable extraction."""

    ROLE_PATTERNS = (
        "machine learning engineer",
        "ml engineer",
        "ai engineer",
        "backend engineer",
        "data engineer",
        "staff engineer",
        "engineering manager",
    )

    EDUCATION_PATTERNS = {
        "bachelor": ("bachelor", "b.tech", "btech", "bs ", "b.s."),
        "master": ("master", "m.tech", "mtech", "ms ", "m.s."),
        "phd": ("phd", "ph.d", "doctorate"),
    }

    REQUIRED_MARKERS = (
        "required",
        "must have",
        "need",
        "needs",
        "mandatory",
        "you have",
        "what you bring",
        "requirements",
        "qualifications",
    )
    PREFERRED_MARKERS = (
        "preferred",
        "nice to have",
        "good to have",
        "bonus",
        "plus",
        "ideally",
        "preferred qualifications",
    )
    DISQUALIFIER_MARKERS = (
        "not a fit",
        "disqualifier",
        "must not",
        "no ",
        "without",
        "only",
    )

    REQUIRED_SKILL_SPECS: tuple[SignalSpec, ...] = (
        SignalSpec("python", ("python", "py"), ("programming", "backend language", "production code"), 0.90),
        SignalSpec("embeddings", ("embedding", "embeddings", "sentence embeddings"), ("vector representation", "semantic representation"), 0.90),
        SignalSpec("retrieval", ("retrieval", "retrieve", "retriever", "rag", "retrieval augmented generation"), ("search recall", "candidate generation", "document search"), 0.88),
        SignalSpec("ranking", ("ranking", "ranker", "reranking", "re-ranking"), ("relevance ordering", "scoring candidates", "search quality"), 0.86),
        SignalSpec("vector_db", ("vector database", "vector databases", "vector db", "vector store", "faiss", "milvus", "pinecone", "weaviate", "qdrant"), ("ann index", "nearest neighbor search", "vector index"), 0.85),
        SignalSpec("production_ml", ("production ml", "ml production", "model serving", "deploy model", "deployed ml", "production deployment"), ("shipping models", "operational ml", "serving systems"), 0.90),
        SignalSpec("evaluation_frameworks", ("evaluation framework", "evaluation frameworks", "eval framework", "offline evaluation"), ("measure retrieval quality", "benchmark search", "quality harness"), 0.82),
        SignalSpec("hybrid_search", ("hybrid search", "keyword and vector", "bm25 and vector", "lexical and semantic"), ("combine sparse dense", "semantic plus keyword"), 0.80),
    )

    PREFERRED_SKILL_SPECS: tuple[SignalSpec, ...] = (
        SignalSpec("lora", ("lora", "low-rank adaptation"), ("parameter efficient finetuning",), 0.72),
        SignalSpec("qlora", ("qlora", "quantized lora"), ("quantized finetuning",), 0.72),
        SignalSpec("peft", ("peft", "parameter efficient fine tuning", "parameter-efficient fine-tuning"), ("adapter tuning",), 0.70),
        SignalSpec("learning_to_rank", ("learning-to-rank", "learning to rank", "ltr"), ("train rankers", "ranking model"), 0.72),
        SignalSpec("hr_tech", ("hr tech", "hrtech", "recruiting tech", "talent tech"), ("hiring workflow", "candidate marketplace"), 0.62),
        SignalSpec("marketplace", ("marketplace", "two-sided marketplace", "talent marketplace"), ("supply demand matching",), 0.60),
        SignalSpec("distributed_systems", ("distributed systems", "distributed system"), ("large distributed service", "fault tolerant systems"), 0.68),
        SignalSpec("large_scale_inference", ("large scale inference", "large-scale inference", "high throughput inference"), ("low latency serving", "batch inference at scale"), 0.68),
        SignalSpec("open_source", ("open source", "oss"), ("public github", "community contribution"), 0.58),
    )

    TOOL_SPECS: tuple[SignalSpec, ...] = (
        SignalSpec("faiss", ("faiss",), ("vector index",), 0.78),
        SignalSpec("milvus", ("milvus",), ("vector database",), 0.78),
        SignalSpec("pinecone", ("pinecone",), ("managed vector database",), 0.74),
        SignalSpec("weaviate", ("weaviate",), ("vector database",), 0.72),
        SignalSpec("qdrant", ("qdrant",), ("vector database",), 0.72),
        SignalSpec("elasticsearch", ("elasticsearch", "elastic search", "opensearch"), ("bm25 search", "lexical search"), 0.70),
        SignalSpec("fastapi", ("fastapi",), ("python api service",), 0.66),
        SignalSpec("aws", ("aws", "sagemaker", "bedrock"), ("cloud deployment",), 0.66),
        SignalSpec("kubernetes", ("kubernetes", "k8s"), ("container orchestration",), 0.66),
    )

    DOMAIN_SPECS: tuple[SignalSpec, ...] = (
        SignalSpec("retrieval_systems", ("retrieval system", "retrieval systems", "search system", "search systems", "rag system"), ("production retrieval", "ranking pipeline"), 0.82),
        SignalSpec("hr_tech", ("hr tech", "hrtech", "recruitment", "recruiting", "talent acquisition"), ("candidate matching", "recruiter workflow"), 0.68),
        SignalSpec("marketplace", ("marketplace", "two-sided marketplace", "talent marketplace"), ("supply demand matching",), 0.64),
        SignalSpec("ml_platform", ("ml platform", "ai platform", "model platform"), ("model serving platform", "ml infrastructure"), 0.66),
    )

    BEHAVIOR_SPECS: tuple[SignalSpec, ...] = (
        SignalSpec("ownership", ("ownership", "own", "end-to-end", "end to end"), ("take responsibility", "drive outcomes"), 0.76),
        SignalSpec("product_mindset", ("product mindset", "product thinking", "customer obsessed", "user impact"), ("solve user problems", "business outcome"), 0.74),
        SignalSpec("startup_mentality", ("startup mentality", "startup", "early stage", "0 to 1", "zero to one"), ("ambiguity", "small team"), 0.72),
        SignalSpec("fast_shipping", ("fast shipping", "ship fast", "iterate quickly", "fast-paced", "fast paced"), ("rapid iteration", "quick experiments"), 0.70),
        SignalSpec("mentoring", ("mentor", "mentoring", "coach", "guide engineers"), ("raise engineering bar",), 0.68),
        SignalSpec("collaboration", ("collaboration", "collaborate", "cross-functional", "work closely"), ("partner with product", "stakeholder alignment"), 0.66),
    )

    EXPERIENCE_SPECS: Mapping[str, SignalSpec] = {
        "production_deployment": SignalSpec(
            "production_deployment",
            ("production deployment", "deployed to production", "deploy models", "model serving", "production ml"),
            ("shipped ml system", "operated model in production"),
            0.82,
        ),
        "recent_coding": SignalSpec(
            "recent_coding",
            ("hands-on coding", "hands-on code", "recent coding", "write code", "coding role", "individual contributor"),
            ("still codes", "day-to-day engineering"),
            0.74,
        ),
        "production_retrieval_systems": SignalSpec(
            "production_retrieval_systems",
            ("production retrieval", "production search", "retrieval systems", "ranking systems", "rag systems"),
            ("search in production", "retrieval at scale"),
            0.84,
        ),
    }

    DISQUALIFIER_SPECS: tuple[SignalSpec, ...] = (
        SignalSpec("research_only", ("research only", "only research", "pure research"), ("papers without shipping",), 0.70),
        SignalSpec("no_production_deployment", ("no production deployment", "without production deployment", "never deployed"), ("no shipped systems",), 0.72),
        SignalSpec("only_recent_llm_experience", ("only recent llm experience", "llm experience only", "just llm prompting"), ("thin llm exposure",), 0.70),
        SignalSpec("no_coding_last_18_months", ("no coding in last 18 months", "not coded in 18 months", "hasn't coded in 18 months"), ("stale hands-on coding",), 0.70),
    )

    EVALUATION_SIGNAL_SPECS: tuple[SignalSpec, ...] = (
        SignalSpec("ndcg", ("ndcg", "normalized discounted cumulative gain"), ("ranking quality metric",), 0.80),
        SignalSpec("map", ("map", "mean average precision"), ("retrieval precision metric",), 0.76),
        SignalSpec("mrr", ("mrr", "mean reciprocal rank"), ("first relevant result",), 0.76),
        SignalSpec("ab_testing", ("a/b testing", "ab testing", "experimentation", "online experiment"), ("online evaluation", "controlled experiment"), 0.72),
        SignalSpec("recruiter_feedback", ("recruiter feedback", "hiring manager feedback", "user feedback"), ("human relevance feedback",), 0.68),
    )

    CULTURE_SPECS: Mapping[str, SignalSpec] = {
        "startup": SignalSpec("startup", ("startup", "early stage", "0 to 1", "zero to one"), ("small team ambiguity",), 0.72),
        "ownership": SignalSpec("ownership", ("ownership", "owner mindset", "own outcomes"), ("accountability",), 0.72),
        "product_engineering": SignalSpec("product_engineering", ("product engineering", "product-minded engineering", "product mindset"), ("build for users",), 0.70),
        "fast_iteration": SignalSpec("fast_iteration", ("fast iteration", "iterate quickly", "ship fast", "fast-paced"), ("rapid experiments",), 0.68),
    }

    def __init__(self, taxonomy: SkillTaxonomy | None = None):
        self.taxonomy = taxonomy or SkillTaxonomy()

    def extract(self, jd_text: str) -> JobDNA:
        text = self._normalize_text(jd_text)
        contexts = self._contexts(text)
        experience_min, experience_max = self._extract_experience(text)
        role = self._extract_role(text)

        required_skills = self._dedupe(
            self._extract_specs(contexts["required"], self.REQUIRED_SKILL_SPECS)
            + self._extract_taxonomy_skills(contexts["required"])
        )
        preferred_skills = self._dedupe(
            self._extract_specs(contexts["preferred"], self.PREFERRED_SKILL_SPECS)
            + self._extract_specs(text, self.PREFERRED_SKILL_SPECS)
            + self._extract_preferred_taxonomy_skills(contexts["preferred"], required_skills)
        )

        required_tools, preferred_tools = self._split_required_preferred(text, contexts, self.TOOL_SPECS)
        required_domains, preferred_domains = self._split_required_preferred(text, contexts, self.DOMAIN_SPECS)
        required_behaviors = self._extract_specs(text, self.BEHAVIOR_SPECS)
        required_experience = self._required_experience(text, experience_min, experience_max)
        disqualifiers = self._extract_disqualifiers(text)
        evaluation_signals = self._extract_specs(text, self.EVALUATION_SIGNAL_SPECS)
        company_culture = self._company_culture(text)

        technical = self._technical_profile(required_skills, required_tools, required_domains)
        career = self._career_profile(text, required_domains, required_experience, company_culture)
        behavior = {behavior: 1.0 for behavior in required_behaviors}
        behavior.update(self._legacy_behavior_profile(required_behaviors))

        return JobDNA(
            role=role,
            experience_min=experience_min,
            experience_max=experience_max,
            required_skills=required_skills,
            preferred_skills=[skill for skill in preferred_skills if skill not in set(required_skills)],
            responsibilities=self._extract_responsibilities(jd_text),
            industries=self._extract_industries(text),
            education=self._extract_education(text),
            required_education=self._extract_education(text),
            work_mode=self._extract_work_mode(text),
            location=self._extract_location(jd_text),
            notice_period_max_days=self._extract_notice_period(text),
            title_keywords=self._title_keywords(role),
            technical=technical,
            career=career,
            behavior=behavior,
            trust={"experience_required": 1.0},
            source_text=jd_text,
            required_tools=required_tools,
            preferred_tools=preferred_tools,
            required_domains=required_domains,
            preferred_domains=preferred_domains,
            required_behaviors=required_behaviors,
            required_experience=required_experience,
            disqualifiers=disqualifiers,
            evaluation_signals=evaluation_signals,
            company_culture=company_culture,
        )

    def _contexts(self, text: str) -> Dict[str, str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        buckets = {"required": [], "preferred": [], "disqualifier": []}
        active_bucket = "required"
        for line in lines:
            bucket = self._line_bucket(line, active_bucket)
            active_bucket = bucket
            buckets[bucket].append(line)
            if bucket == "required":
                buckets["required"].append(self._next_clause(line))
        return {key: "\n".join(value) for key, value in buckets.items()}

    def _line_bucket(self, line: str, active_bucket: str) -> str:
        marker_groups = (
            ("disqualifier", self.DISQUALIFIER_MARKERS),
            ("preferred", self.PREFERRED_MARKERS),
            ("required", self.REQUIRED_MARKERS),
        )
        return next((name for name, markers in marker_groups if self._has_any(line, markers)), active_bucket)

    def _split_required_preferred(
        self,
        text: str,
        contexts: Mapping[str, str],
        specs: Sequence[SignalSpec],
    ) -> tuple[List[str], List[str]]:
        required = self._extract_specs(contexts["required"], specs)
        preferred = self._extract_specs(contexts["preferred"], specs)
        preferred = self._dedupe(preferred + [
            label
            for label in self._extract_specs(text, specs)
            if label not in set(required) and self._near_preferred_marker(text, label, specs)
        ])
        return required, preferred

    def _extract_taxonomy_skills(self, text: str) -> List[str]:
        found = [
            canonical
            for alias, canonical in self.taxonomy.aliases.items()
            if self._contains_term(text, alias)
        ]
        return sorted(set(found), key=lambda skill: self.taxonomy.weight_for(skill), reverse=True)

    def _extract_preferred_taxonomy_skills(self, text: str, required: Iterable[str]) -> List[str]:
        preferred_text = self._section_after(text, self.PREFERRED_MARKERS)
        required_set = set(required)
        return [skill for skill in self._extract_taxonomy_skills(preferred_text) if skill not in required_set]

    def _extract_specs(self, text: str, specs: Sequence[SignalSpec]) -> List[str]:
        matches = [
            spec.label
            for spec in specs
            if self._matches_spec(text, spec)
        ]
        return self._dedupe(matches)

    def _matches_spec(self, text: str, spec: SignalSpec) -> bool:
        return self._matches_alias(text, spec.aliases) or self._semantic_match(text, spec)

    def _matches_alias(self, text: str, aliases: Iterable[str]) -> bool:
        return any(self._contains_term(text, alias) for alias in aliases)

    def _semantic_match(self, text: str, spec: SignalSpec) -> bool:
        text_tokens = self._tokens(text)
        cue_scores = [
            len(text_tokens & self._tokens(cue)) / max(len(self._tokens(cue)), 1)
            for cue in spec.cues
        ]
        return bool(cue_scores and max(cue_scores) >= spec.weight)

    def _required_experience(self, text: str, minimum: int, maximum: int) -> Dict[str, object]:
        experience = {
            "years": {"min": minimum, "max": maximum, "label": f"{minimum}-{maximum} years"},
        }
        experience.update({
            key: self._matches_spec(text, spec)
            for key, spec in self.EXPERIENCE_SPECS.items()
        })
        return experience

    def _extract_disqualifiers(self, text: str) -> List[str]:
        explicit = self._extract_specs(text, self.DISQUALIFIER_SPECS)
        contextual = [
            spec.label
            for spec in self.DISQUALIFIER_SPECS
            if self._matches_spec(self._section_after(text, self.DISQUALIFIER_MARKERS), spec)
        ]
        return self._dedupe(explicit + contextual)

    def _company_culture(self, text: str) -> Dict[str, object]:
        return {
            key: {
                "required": self._matches_spec(text, spec),
                "evidence": self._evidence_for(text, spec),
            }
            for key, spec in self.CULTURE_SPECS.items()
            if self._matches_spec(text, spec)
        }

    def _technical_profile(
        self,
        required_skills: Iterable[str],
        required_tools: Iterable[str],
        required_domains: Iterable[str],
    ) -> Dict[str, float]:
        weighted = {
            skill: self.taxonomy.weight_for(skill)
            for skill in required_skills
        }
        weighted.update({tool: self.taxonomy.weight_for(tool) for tool in required_tools})
        weighted.update({domain: 0.60 for domain in required_domains})
        return weighted

    def _career_profile(
        self,
        text: str,
        domains: Iterable[str],
        experience: Mapping[str, object],
        culture: Mapping[str, object],
    ) -> Dict[str, float]:
        signals = {
            "production_ml": float(bool(experience.get("production_deployment")) or self._has_any(text, ("production ml", "model serving"))),
            "backend": float(self._has_any(text, ("backend", "api", "fastapi", "service"))),
            "big_data": float(self._has_any(text, ("spark", "kafka", "airflow", "data pipeline"))),
            "leadership": float(self._has_any(text, ("lead", "mentor", "manage", "staff engineer"))),
            "management": float(self._has_any(text, ("manager", "people management", "hiring"))),
            "ai_research": float(self._has_any(text, ("research", "paper", "publication", "novel model"))),
            "mlops": float(self._has_any(text, ("mlops", "ci/cd", "monitoring", "model registry"))),
            "cloud": float(self._has_any(text, ("aws", "gcp", "azure", "cloud", "kubernetes"))),
            "startup": float("startup" in culture),
            "production_retrieval_systems": float("retrieval_systems" in set(domains) or bool(experience.get("production_retrieval_systems"))),
        }
        return signals

    def _legacy_behavior_profile(self, behaviors: Iterable[str]) -> Dict[str, float]:
        behavior_set = set(behaviors)
        return {
            "ship_fast": float("fast_shipping" in behavior_set),
            "ownership": float("ownership" in behavior_set),
            "communication": float("collaboration" in behavior_set),
        }

    def _extract_experience(self, text: str) -> tuple[int, int]:
        range_patterns = (
            r"(\d+)\s*(?:to|-|–)\s*(\d+)\s*(?:years?|yrs?)",
            r"(\d+)\s*(?:\+)\s*(?:years?|yrs?)",
        )
        for pattern in range_patterns:
            match = re.search(pattern, text)
            if match:
                minimum = int(match.group(1))
                maximum = int(match.group(2)) if match.lastindex and match.lastindex >= 2 and match.group(2) else max(minimum + 4, minimum)
                return minimum, maximum
        single = re.search(r"(\d+)\s*\+?\s*(?:years?|yrs?)", text)
        if single:
            minimum = int(single.group(1))
            return minimum, max(minimum + 4, minimum)
        return 0, 50

    def _extract_role(self, text: str) -> str:
        role = next((role for role in self.ROLE_PATTERNS if role in text), "")
        if role:
            return role.title()
        title_match = re.search(r"(?:role|title)\s*[:\-]\s*(.+)", text)
        return title_match.group(1).strip().title() if title_match else "Open Role"

    def _extract_responsibilities(self, jd_text: str) -> List[str]:
        lines = [line.strip(" -\t") for line in jd_text.splitlines()]
        return [line for line in lines if len(line.split()) >= 5][:12]

    def _extract_industries(self, text: str) -> List[str]:
        industries = ("ai", "saas", "fintech", "healthcare", "ecommerce", "recruitment", "hr tech", "marketplace")
        return [industry for industry in industries if self._contains_term(text, industry)]

    def _extract_education(self, text: str) -> List[str]:
        return [
            degree
            for degree, aliases in self.EDUCATION_PATTERNS.items()
            if any(self._contains_term(text, alias) for alias in aliases)
        ]

    def _extract_work_mode(self, text: str) -> str:
        mode_patterns = {
            "remote": (r"\bremote\b", r"\bwork from home\b"),
            "hybrid": (r"\bhybrid\s+(?:work|role|mode|position)\b",),
            "onsite": (r"\bonsite\b", r"\bon-site\b", r"\bin office\b"),
        }
        return next(
            (
                mode
                for mode, patterns in mode_patterns.items()
                if any(re.search(pattern, text) for pattern in patterns)
            ),
            "",
        )

    def _extract_location(self, jd_text: str) -> str:
        match = re.search(r"(?:location|based in)\s*[:\-]\s*([^\n,]+(?:,\s*[^\n]+)?)", jd_text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_notice_period(self, text: str) -> int:
        match = re.search(r"notice(?: period)?\D{0,12}(\d+)\s*days?", text)
        return int(match.group(1)) if match else 90

    def _title_keywords(self, role: str) -> List[str]:
        normalized = role.lower()
        keywords = [part for part in re.split(r"\W+", normalized) if len(part) > 2]
        if "ml" in normalized or "machine learning" in normalized:
            keywords.extend(["machine learning", "ml", "ai"])
        return sorted(set(keywords))

    def _near_preferred_marker(self, text: str, label: str, specs: Sequence[SignalSpec]) -> bool:
        spec = next((item for item in specs if item.label == label), None)
        if not spec:
            return False
        alias_positions = [text.find(alias) for alias in spec.aliases if alias in text]
        marker_positions = [text.find(marker) for marker in self.PREFERRED_MARKERS if marker in text]
        return any(abs(alias - marker) <= 180 for alias in alias_positions if alias >= 0 for marker in marker_positions if marker >= 0)

    def _evidence_for(self, text: str, spec: SignalSpec) -> List[str]:
        return [
            alias
            for alias in spec.aliases
            if self._contains_term(text, alias)
        ][:3]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[ \t]+", " ", text.lower())

    @staticmethod
    def _next_clause(line: str) -> str:
        parts = re.split(r"[:\-]", line, maxsplit=1)
        return parts[-1] if parts else line

    @staticmethod
    def _section_after(text: str, markers: Iterable[str]) -> str:
        positions = [text.find(marker) for marker in markers if marker in text]
        return text[min(positions):] if positions else ""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}

    @staticmethod
    def _has_any(text: str, terms: Iterable[str]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        escaped = re.escape(term.lower().strip())
        escaped = escaped.replace(r"\ ", r"[\s\-_\/]+")
        return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text))

    @staticmethod
    def _dedupe(values: Iterable[str]) -> List[str]:
        return list(dict.fromkeys(value for value in values if value))
