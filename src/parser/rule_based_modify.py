"""
VM-AI - Rule-based Modify Mode Parser
Handles importance/difficulty changes with explicit keyword mappings.

Written by: Vanea
"""

import re
from typing import Dict, Optional

NEGATION_WORDS = {"not", "no", "never", "without"}

def _is_negated(s: str, keyword: str) -> bool:
    """Check if keyword in s is preceded by a negation word within 3 tokens."""
    idx = s.find(keyword)
    if idx < 0:
        return False
    before = s[:idx].strip()
    if not before:
        return False
    tokens = before.split()
    lookback = tokens[-3:]
    for token in lookback:
        if token in NEGATION_WORDS or token.endswith("n't"):
            return True
    return False


def parse_modify_rule_based(
    change_prompt: str, existing_task: Optional[Dict] = None
) -> Dict:
    """Parse modify instruction using rules for importance/difficulty.

    Args:
        change_prompt: The change instruction (e.g., "make it urgent")
        existing_task: Optional existing task (not used in rule-based mode)

    Returns:
        Dict with changed fields: {"field": {"value": ..., "predicted": ...}}
    """
    s = change_prompt.lower().strip()
    result = {}

    importance_map = {
        "urgent": 0.95,
        "critical": 0.98,
        "asap": 0.95,
        "very important": 0.9,
        "very urgent": 0.95,
        "important": 0.75,
        "high priority": 0.85,
        "top priority": 0.95,
        "low priority": 0.15,
        "not urgent": 0.2,
        "optional": 0.15,
        "can wait": 0.15,
        "minor": 0.15,
        "whenever": 0.1,
    }

    difficulty_map = {
        "hard": 0.85,
        "difficult": 0.85,
        "challenging": 0.8,
        "complex": 0.75,
        "intense": 0.9,
        "heavy": 0.85,
        "tough": 0.75,
        "easier": 0.15,
        "easy": 0.15,
        "simple": 0.15,
        "light": 0.2,
        "quick": 0.2,
        "moderate": 0.45,
        "medium": 0.45,
    }

    for keyword in sorted(importance_map.keys(), key=len, reverse=True):
        if keyword in s and not _is_negated(s, keyword):
            result["importance"] = {
                "value": str(importance_map[keyword]),
                "predicted": False,
            }
            break

    for keyword in sorted(difficulty_map.keys(), key=len, reverse=True):
        if keyword in s and not _is_negated(s, keyword):
            result["difficulty"] = {
                "value": str(difficulty_map[keyword]),
                "predicted": False,
            }
            break

    if "cancel" in s or "remove" in s or "clear" in s:
        if "fixed" in s or "time" in s:
            result["fixed_time"] = {"value": False, "predicted": False}
            result["fixed_start"] = {"value": None, "predicted": False}
        if "recurr" in s:
            result["recurrent"] = {"value": False, "predicted": False}
            result["recurrence_days"] = {"value": None, "predicted": False}

    time_match = re.search(r"at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))", s)
    if time_match and not _is_negated(s, "at"):
        from rule_based_add import normalize_time

        normalized = normalize_time(time_match.group(1))
        if normalized:
            result["fixed_time"] = {"value": True, "predicted": False}
            result["fixed_start"] = {"value": normalized, "predicted": False}

    deadline_match = re.search(
        r"(?:push|change|move)\s+(?:the\s+)?deadline\s+to\s+(\w+)", s
    )
    if not deadline_match:
        deadline_match = re.search(r"deadline\s+(?:to|is)\s+(\w+)", s)
    if not deadline_match:
        deadline_match = re.search(r"due\s+(?:by\s+)?(\w+)", s)
    if deadline_match and not _is_negated(s, deadline_match.group(0)):
        result["deadline"] = {
            "value": deadline_match.group(1).title(),
            "predicted": False,
        }

    start_match = re.search(r"(?:start|begin|kick off)\s+(?:on\s+)?(\w+)", s)
    if start_match and not _is_negated(s, start_match.group(0)):
        result["start"] = {"value": start_match.group(1).title(), "predicted": False}

    duration_match = re.search(r"(\d+)\s*(?:minute|min|hour|hr)s?", s)
    if duration_match and not _is_negated(s, duration_match.group(0)):
        v = int(duration_match.group(1))
        if "hour" in s:
            v *= 60
        result["duration"] = {"value": str(v), "predicted": False}

    category_map = {
        "work": "work",
        "job": "work",
        "fitness": "fitness",
        "exercise": "fitness",
        "workout": "fitness",
        "health": "health",
        "medical": "health",
        "study": "study",
        "learning": "study",
        "home": "home",
        "house": "home",
        "finance": "finance",
        "money": "finance",
        "shopping": "shopping",
        "travel": "travel",
        "trip": "travel",
        "creative": "creative",
        "art": "creative",
        "family": "family",
        "kids": "family",
        "social": "social",
        "friend": "social",
        "personal": "personal",
    }
    for keyword, cat in category_map.items():
        if keyword in s and not _is_negated(s, keyword):
            result["category"] = {"value": cat, "predicted": True}
            break

    location_map = {
        "home": "home",
        "office": "office",
        "work": "work",
        "gym": "gym",
        "library": "library",
        "coffee shop": "coffee shop",
        "cafe": "coffee shop",
        "park": "park",
        "outdoor": "park",
        "school": "school",
        "university": "school",
        "supermarket": "supermarket",
        "grocery": "supermarket",
    }
    for keyword, loc in location_map.items():
        if keyword in s and not _is_negated(s, keyword):
            result["location"] = {"value": loc, "predicted": False}
            break

    return result


if __name__ == "__main__":
    tests = [
        "make it urgent",
        "make it critical",
        "make it optional",
        "make it hard",
        "make it easier",
        "change deadline to friday",
        "set time to 3pm",
        "make it high priority and hard",
    ]

    for t in tests:
        result = parse_modify_rule_based(t)
        print(f"\n'{t}'")
        for field, entry in result.items():
            print(
                f"  {field}: {entry['value']} ({'EXP' if not entry['predicted'] else 'PRD'})"
            )
