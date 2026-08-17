"""Pydantic schemas for the audit API."""

from typing import Any

from pydantic import BaseModel, Field


class AuditRequest(BaseModel):
    question: str = Field(min_length=1)
    model: str
    attack_vector: str
    options: dict[str, str] | None = None
    correct_label: str | None = None


class AuditResponse(BaseModel):
    run_id: str
    sycophancy_detected: bool
    nist_report: dict[str, Any]
