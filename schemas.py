"""Pydantic schemas used by the Helpdesk API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    ticket_id: Optional[int] = None


class AnalyzeRequest(BaseModel):
    issue: str = Field(min_length=1)
    branch_name: Optional[str] = None
    history: list[ChatMessage] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    intent: Literal["GREETING_OR_THANKS", "FOLLOWUP_CHAT", "IT_ISSUE"] = "IT_ISSUE"
    chat_reply: Optional[str] = None
    sop_found: bool = False
    sop_solution: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
    category: Literal["HARDWARE", "SOFTWARE"] = "HARDWARE"
    routing_target: str = "IT Infrastruktur / Vendor"
    reason: str = ""
