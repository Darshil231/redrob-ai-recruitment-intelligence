# Redrob AI Recruitment Intelligence Platform

Production-grade candidate ranking for recruiter workflows. The platform parses a job description, filters obvious mismatches, ranks candidates with explainable fast scoring, applies semantic re-ranking only to the shortlist, and returns structured hiring recommendations with CSV and JSON export support.

✨ Features
Process 100,000 candidate profiles
Multi-stage AI ranking pipeline
Job DNA Extraction
Eligibility Filtering
Skill Matching
Career Analysis
Behavioral Signal Analysis
Role Relevance Scoring
Semantic AI Matching (Sentence Transformers)
Explainable AI Recommendations
FastAPI REST API
CSV & JSON Export

## Architecture

```text
Job Description
  -> Job DNA Extraction
  -> Candidate Parsing
  -> Eligibility Filter
  -> Skill Matching
  -> Career Analysis
  -> Fast Ranking
  -> Top 200
  -> Semantic Embedding Ranking
  -> Top 50
  -> LLM-style Structured Reasoning
  -> Top 10
  -> CSV Export / JSON API
```

## Module Map

- `src/core`: Typed domain models and configuration.
- `src/parsers`: Candidate JSON/JSONL parser and JD text/DOCX parser.
- `src/intelligence`: Job DNA, eligibility filtering, skill matching, career analysis, semantic matching, and structured reasoning.
- `src/ranking`: Ranking formulas and end-to-end pipeline orchestration.
- `src/services`: Reusable application services for candidates and ranking.
- `src/export`: CSV and JSON exporters.
- `src/api`: FastAPI server for ranking and candidate access.
- `tests`: Unit tests for parser, DNA, filters, semantic, ranking, and pipeline.

## 📂 Project Structure

```text
redrob-ai-recruitment-intelligence/
├── src/
│   ├── api/
│   ├── core/
│   ├── export/
│   ├── intelligence/
│   ├── parsers/
│   ├── ranking/
│   ├── services/
│   └── utils/
├── data/
├── tests/
├── exports/
├── app.py
├── requirements.txt
├── README.md
└── submission.csv
```

## Tech Stack
Python 3.11
FastAPI
Sentence Transformers
PyTorch
NumPy
Pydantic
Pytest
python-docx
JSON / JSONL / CSV

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

`python-docx` is required for `.docx` JD parsing. Text JDs work without it.

## CLI Usage

```bash
python app.py
```

Outputs:

- `exports/top_candidates.csv`
- `exports/top_candidates.json`

## API Usage

```bash
uvicorn src.api.server:app --reload
```

Endpoints:

- `GET /health`
- `POST /upload-jd`
- `POST /rank`
- `GET /candidate/{id}`
- `GET /top`

Example `POST /rank` body:

```json
{
  "jd_text": "Role: Senior ML Engineer. Required skills: Python, LLM, RAG, FastAPI.",
  "candidates": []
}
```

> **Note:**  
> The full `data/candidates.jsonl` dataset is not included in this repository because it exceeds GitHub's 100 MB file size limit. The repository includes `data/sample_candidates.json` for testing and development. To run the complete pipeline on the full dataset, place the official `candidates.jsonl` file in the `data/` directory.

## Scoring

Fast score:

```text
40% Skill + 30% Career + 30% Eligibility Filters
```

Final score:

```text
40% Fast Score + 40% Semantic Score + 20% Reasoning Confidence
```

Every ranked candidate includes matched skills, missing skills, relevant experience, relevant projects, selection reasons, rejection risks, recommendation, and confidence.

## Verification

```bash
python -m pytest -q
```

For quick syntax verification:

```bash
python -m compileall app.py src tests
```
