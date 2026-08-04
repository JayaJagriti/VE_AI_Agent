"""
enums.py

Shared enum types used across client_schema.py and requirement_schema.py.
Kept in one place so the agent's question logic and the schemas always
agree on the same set of allowed values.
"""

from enum import Enum


class TechnicalLevel(str, Enum):
    """How technical the client themselves is — drives the agent's question style."""
    TECHNICAL = "technical"
    NON_TECHNICAL = "non_technical"
    UNKNOWN = "unknown"


class RoleCategory(str, Enum):
    """The domains Virtual Employee actually staffs for (from their service lines)."""
    SOFTWARE_DEVELOPMENT = "software_development"
    WEB_DESIGN_DEVELOPMENT = "web_design_development"
    MOBILE_APP_DEVELOPMENT = "mobile_app_development"
    DATA_SCIENCE_AI = "data_science_ai"
    QA_TESTING = "qa_testing"
    GRAPHIC_DESIGN = "graphic_design"
    DIGITAL_MARKETING_SEO = "digital_marketing_seo"
    CONTENT_WRITING = "content_writing"
    ACCOUNTING_BOOKKEEPING = "accounting_bookkeeping"
    DATA_ENTRY = "data_entry"
    ADMIN_VIRTUAL_ASSISTANT = "admin_virtual_assistant"
    CUSTOMER_SUPPORT = "customer_support"
    RECRUITMENT_HR = "recruitment_hr"
    LEGAL_PROCESS = "legal_process"
    OTHER = "other"


class EngagementType(str, Enum):
    FULL_TIME = "full_time"          # 8 hrs/day, 5 days/week dedicated resource — VE's core model
    PART_TIME = "part_time"
    PROJECT_BASED = "project_based"
    UNKNOWN = "unknown"


class ExperienceLevel(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    UNSPECIFIED = "unspecified"


class UrgencyLevel(str, Enum):
    IMMEDIATE = "immediate"       # within a week
    SHORT_TERM = "short_term"     # within a month
    FLEXIBLE = "flexible"
    UNSPECIFIED = "unspecified"


class RequirementStatus(str, Enum):
    DRAFT = "draft"                 # agent is still gathering info
    PENDING_REVIEW = "pending_review"  # all fields captured, awaiting summary generation
    COMPLETE = "complete"           # summary generated and delivered
