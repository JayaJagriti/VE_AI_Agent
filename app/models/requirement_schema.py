"""
requirement_schema.py

Pydantic models for everything the agent captures during a conversation.

Per review: client identity and project requirement are merged into one
`RequirementState` model for now, since in practice they're filled in
together, turn by turn, from the same chat session. If client profiles
ever need to be reused across multiple separate requirements, split
`RequirementState` back into a `ClientProfile` + `RequirementData` pair —
the field groupings below are kept together to make that split easy later.

No logic lives here — only data shape and validation rules. This is the
contract that:
  - ConversationManager fills in field by field during the chat
  - app/memory/state_store.py persists
  - app/agent/summary_generator.py reads from to produce the final summary
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, EmailStr, Field, field_validator


declined_fields: List[str] = Field(default_factory=list)

from app.models.enums import (
    EngagementType,
    ExperienceLevel,
    RequirementStatus,
    RoleCategory,
    TechnicalLevel,
    UrgencyLevel,
)


class SkillRequirement(BaseModel):
    """A single skill/technology the client needs the hired resource to have."""

    skill_name: str
    is_mandatory: bool = True
    years_experience: Optional[int] = Field(default=None, ge=0, le=40)


class BudgetRange(BaseModel):
    """Client's stated or estimated budget for the engagement."""

    min_amount: Optional[float] = Field(default=None, ge=0)
    max_amount: Optional[float] = Field(default=None, ge=0)
    currency: str = Field(default="USD", description="ISO currency code, e.g. USD, GBP, INR")
    period: str = Field(
        default="monthly", description="'monthly', 'hourly', or 'project_total'"
    )

    @field_validator("max_amount")
    @classmethod
    def max_not_less_than_min(cls, v, info):
        min_amount = info.data.get("min_amount")
        if v is not None and min_amount is not None and v < min_amount:
            raise ValueError("max_amount cannot be less than min_amount")
        return v


class RequirementState(BaseModel):
    """The single state object tracked per conversation session — combines
    who the client is with what they need. Filled in incrementally, so
    almost every field is optional except session_id."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))

    # --- Client identity (formerly ClientProfile) ---
    client_name: Optional[str] = None
    client_email: Optional[EmailStr] = None
    client_phone: Optional[str] = None
    company_name: Optional[str] = None
    company_industry: Optional[str] = None
    country: Optional[str] = None
    technical_level: TechnicalLevel = Field(
        default=TechnicalLevel.UNKNOWN,
        description="Inferred from conversation — drives how the agent phrases questions.",
    )
    technical_level_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    preferred_contact_method: Optional[str] = None

    # --- Requirement details (formerly RequirementData) ---
    role_category: RoleCategory = RoleCategory.OTHER
    role_title: Optional[str] = Field(
        default=None, description="Client's own words, e.g. 'Senior React Developer'"
    )
    required_skills: List[SkillRequirement] = Field(default_factory=list)
    experience_level: ExperienceLevel = ExperienceLevel.UNSPECIFIED
    engagement_type: EngagementType = EngagementType.UNKNOWN
    number_of_resources: Optional[int] = Field(default=None, ge=1)
    estimated_budget: Optional[BudgetRange] = None
    urgency: UrgencyLevel = UrgencyLevel.UNSPECIFIED
    preferred_start_date: Optional[datetime] = None
    required_timezone_overlap: Optional[str] = None
    project_description: Optional[str] = None
    additional_notes: Optional[str] = None

    # --- Bookkeeping ---
    status: RequirementStatus = RequirementStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class RequirementSummary(BaseModel):
    """The final generated summary handed back to the client / sales team."""

    session_id: str
    summary_text: str = Field(..., description="Human-readable narrative summary")
    key_points: List[str] = Field(default_factory=list)
    recommended_next_step: Optional[str] = Field(
        default=None, description="e.g. 'Schedule a call with a VE account manager'"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)
