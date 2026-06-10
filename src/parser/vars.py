"""
VM-AI - Global Variables and Constants
Stores dataset names, field definitions, and shared constants.

Written by: Vanea
"""

PARSER_MODEL_NAME = "finetuned_parser"
SYNTHETIC_DATASET = "VMAI_SYNTHETIC_Data.yaml"
REAL_DATASET = "VMAI_REAL_Data.yaml"
SPECIFIC_DATASET = "VMAI_SPECIFIC_Data.yaml"

EXP = "EXP"
PRD = "PRD"

PREDICTED_FIELDS = {
    "difficulty",
    "duration",
    "category",
    "location",
    "importance",
    "start",
    "deadline",
    "fixed_time",
    "fixed_start",
    "recurrent",
    "recurrence_days",
}

ALWAYS_EXPLICIT = {"name"}

ALL_FIELDS = {
    "name": None,
    "start": None,
    "deadline": None,
    "difficulty": None,
    "duration": None,
    "category": None,
    "location": None,
    "importance": None,
    "fixed_time": False,
    "fixed_start": None,
    "recurrent": False,
    "recurrence_days": None,
}

TRACKED_FIELDS = [
    "name",
    "start",
    "deadline",
    "difficulty",
    "importance",
    "duration",
    "category",
    "location",
    "fixed_time",
    "fixed_start",
    "recurrent",
    "recurrence_days",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

FIELD_MAP = {
    "TASK": "name",
    "DEADLINE": "deadline",
    "DATE": "start",
    "TIME": "fixed_start",
    "DURATION": "duration",
    "LOCATION": "location",
    "PRIORITY": "importance",
    "DIFFICULTY": "difficulty",
    "CATEGORY": "category",
}

VALID_CATEGORIES = {
    "work",
    "study",
    "fitness",
    "health",
    "personal",
    "finance",
    "home",
    "family",
    "social",
    "errands",
    "travel",
    "creative",
    "learning",
    "admin",
    "shopping",
}
