"""FastAPI application exposing recruitment ranking workflows."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.config import AppConfig
from src.parsers.candidate_parser import CandidateParser
from src.services.candidate_service import CandidateService
from src.services.ranking_service import RankingService


class RankRequest(BaseModel):
    jd_text: str = Field(min_length=20)
    candidates: list[dict[str, Any]] | None = None


class UploadJDRequest(BaseModel):
    jd_text: str = Field(min_length=20)


config = AppConfig()
candidate_service = CandidateService(config.candidates_path)
ranking_service = RankingService(config=config)
latest_jd_text = ""

app = FastAPI(
    title="Redrob AI Recruitment Intelligence API",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload-jd")
def upload_jd(request: UploadJDRequest) -> dict[str, int | str]:
    global latest_jd_text
    latest_jd_text = request.jd_text
    return {"status": "accepted", "characters": len(latest_jd_text)}


@app.post("/rank")
def rank(request: RankRequest) -> list[dict[str, Any]]:
    candidates = (
        CandidateParser.from_records(request.candidates)
        if request.candidates is not None
        else candidate_service.list_candidates()
    )
    results = ranking_service.rank(request.jd_text, candidates)
    return [result.to_dict() for result in results]


@app.get("/candidate/{candidate_id}")
def candidate(candidate_id: str) -> dict[str, Any]:
    found = candidate_service.get_candidate(candidate_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return found.to_dict()


@app.get("/top")
def top() -> list[dict[str, Any]]:
    return [result.to_dict() for result in ranking_service.latest()]
