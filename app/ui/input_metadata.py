from app.models.enums import ExperienceLevel, EngagementType, UrgencyLevel

INPUT_METADATA = {
    "experience_level": {
        "type": "choice",
        "options": [
            {"label": "Junior", "value": ExperienceLevel.JUNIOR.value},
            {"label": "Mid", "value": ExperienceLevel.MID.value},
            {"label": "Senior", "value": ExperienceLevel.SENIOR.value},
            {"label": "Lead", "value": ExperienceLevel.LEAD.value},
        ],
    },
    "engagement_type": {
        "type": "choice",
        "options": [
            {"label": "Full-time", "value": EngagementType.FULL_TIME.value},
            {"label": "Part-time", "value": EngagementType.PART_TIME.value},
            {"label": "Project-based", "value": EngagementType.PROJECT_BASED.value},
        ],
    },
    "urgency": {
        "type": "choice",
        "options": [
            {"label": "Immediately", "value": UrgencyLevel.IMMEDIATE.value},
            {"label": "Within a month", "value": UrgencyLevel.SHORT_TERM.value},
            {"label": "Flexible", "value": UrgencyLevel.FLEXIBLE.value},
        ],
    },
}