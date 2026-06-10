"""
VM-AI - Rule-based Add Mode Parser (MVP)
Extracts basic fields from natural language without a model.

Written by: Vanea
"""

import re

from vars import ALL_FIELDS, DAYS


def normalize_time(time_str):
    if not time_str:
        return None
    time_str = str(time_str).strip().lower()
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", time_str)
    if m:
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        if m.group(3) == "pm" and h != 12:
            h += 12
        elif m.group(3) == "am" and h == 12:
            h = 0
        return f"{h:02d}:{mi:02d}"
    if "morning" in time_str:
        return "08:00"
    if "afternoon" in time_str:
        return "13:00"
    if "evening" in time_str:
        return "18:00"
    if "noon" in time_str:
        return "12:00"
    if "midnight" in time_str:
        return "00:00"
    return None


def parse_add(sentence: str) -> dict:
    s = sentence.lower().strip()
    schema = {f: {"value": ALL_FIELDS[f], "predicted": True} for f in ALL_FIELDS}
    schema["name"]["predicted"] = False
    schema["fixed_time"]["predicted"] = False
    schema["fixed_start"]["predicted"] = False
    schema["recurrent"]["predicted"] = False
    schema["recurrence_days"]["predicted"] = False

    schema["difficulty"]["value"] = "0.5"
    schema["importance"]["value"] = "0.5"
    schema["category"]["value"] = "personal"
    schema["duration"]["value"] = "30"
    schema["location"]["value"] = None

    name = s
    for prefix in [
        "i need to ",
        "i have to ",
        "i want to ",
        "schedule ",
        "set ",
        "create ",
        "remind me to ",
    ]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    if "every " in s or "daily" in s:
        for kw in [" every ", " daily "]:
            idx = name.find(kw)
            if idx > 0:
                name = name[:idx].strip()
    else:
        for kw in [
            " at ",
            " by ",
            " for ",
            " from ",
            " with ",
            " to ",
            " - ",
            " — ",
            " not ",
            " never ",
        ]:
            idx = name.find(kw)
            if idx > 0:
                name = name[:idx]
                break
        if "n't " in name:
            idx = name.find("n't ")
            if idx > 0:
                name = name[:idx]
                # find the word boundary before the contraction
                prev_space = name.rfind(" ", 0, idx - 1)
                if prev_space > 0:
                    name = name[:prev_space]
    name = re.sub(r"^\d+\s*(minute|min|hour|hr)s?\s*", "", name)
    name = re.sub(
        r"^(hard|difficult|easy|simple|quick|moderate|light|challenging)\s+", "", name
    )
    name = re.sub(r"^(urgent|important|critical|optional)\s+", "", name)
    name = re.sub(r"^(not|no|never)\s+", "", name)
    schema["name"]["value"] = name.strip()

    tm = re.search(r"at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))", s)
    if tm:
        n = normalize_time(tm.group(1))
        if n:
            schema["fixed_time"]["value"] = True
            schema["fixed_start"]["value"] = n

    dm = re.search(r"(\d+(?:\.\d+)?)\s*(?:minute|min|hour|hr)s?", s)
    if dm:
        v = float(dm.group(1))
        if "hour" in s or "hr" in s:
            v *= 60
        schema["duration"]["value"] = str(int(v))
        schema["duration"]["predicted"] = False
    else:
        duration_map = {
            "gym": 45,
            "workout": 45,
            "yoga": 45,
            "run": 30,
            "jog": 30,
            "stretch": 15,
            "call": 15,
            "call mom": 15,
            "call dad": 15,
            "meeting": 30,
            "standup": 15,
            "presentation": 60,
            "report": 60,
            "study": 60,
            "homework": 45,
            "code": 60,
            "coding": 60,
            "grocery": 45,
            "shopping": 45,
            "rent": 10,
            "bill": 10,
            "pay": 10,
            "laundry": 45,
            "clean": 45,
            "cook": 30,
            "flight": 30,
            "book flight": 30,
            "kids": 15,
            "children": 15,
            "pick up": 15,
            "guitar": 30,
            "practice": 30,
            "blog": 45,
            "meditat": 15,
        }
        for task, dur in duration_map.items():
            if task in s:
                schema["duration"]["value"] = str(dur)
                schema["duration"]["predicted"] = True
                break

    if "every " in s or "daily" in s or "each " in s:
        schema["recurrent"]["value"] = True
        days = [d for d in DAYS if d.lower() in s]
        if not days:
            if "weekday" in s:
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            elif "daily" in s or "every day" in s:
                days = DAYS[:]
        if days:
            schema["recurrence_days"]["value"] = ",".join(days)

    for kw, val in [
        ("hard", 0.85),
        ("difficult", 0.85),
        ("challenging", 0.8),
        ("tough", 0.75),
        ("easy", 0.15),
        ("simple", 0.15),
        ("light", 0.2),
        ("quick", 0.2),
        ("moderate", 0.45),
    ]:
        if kw in s:
            schema["difficulty"]["value"] = str(val)
            schema["difficulty"]["predicted"] = False
            break

    for kw, val in [
        ("urgent", 0.95),
        ("critical", 0.95),
        ("asap", 0.95),
        ("very important", 0.9),
        ("important", 0.75),
        ("low priority", 0.15),
        ("not urgent", 0.2),
        ("optional", 0.2),
    ]:
        if kw in s:
            schema["importance"]["value"] = str(val)
            schema["importance"]["predicted"] = False
            break
    else:
        importance_map = {
            "rent": 0.8,
            "bill": 0.8,
            "pay rent": 0.8,
            "tax": 0.85,
            "kids": 0.8,
            "children": 0.8,
            "pick up kids": 0.8,
            "doctor": 0.8,
            "presentation": 0.7,
            "meeting": 0.6,
            "standup": 0.5,
            "exam": 0.8,
            "homework": 0.7,
        }
        for task, imp in importance_map.items():
            if task in s:
                schema["importance"]["value"] = str(imp)
                schema["importance"]["predicted"] = True
                break

    if schema["difficulty"]["value"] == "0.5":
        difficulty_map = {
            "bug": 0.7,
            "crash": 0.8,
            "fix": 0.6,
            "tax": 0.7,
            "rent": 0.3,
            "bill": 0.3,
            "exam": 0.7,
            "study": 0.6,
            "presentation": 0.6,
            "report": 0.5,
            "shopping": 0.2,
            "grocery": 0.2,
        }
        for task, diff in difficulty_map.items():
            if task in s:
                schema["difficulty"]["value"] = str(diff)
                schema["difficulty"]["predicted"] = True
                break

    if "tomorrow" in s:
        schema["deadline"]["value"] = "tomorrow"
    elif "next week" in s:
        schema["deadline"]["value"] = "next week"
    else:
        for d in DAYS:
            if d.lower() in s and "every" not in s.split(d.lower())[0][-6:]:
                schema["deadline"]["value"] = d
                break

    for kw, loc in [
        ("at the library", "library"),
        ("at the gym", "gym"),
        ("at the office", "office"),
        ("at home", "home"),
        ("from home", "home"),
        ("at the coffee shop", "coffee shop"),
        ("at the supermarket", "supermarket"),
    ]:
        if kw in s:
            schema["location"]["value"] = loc
            schema["location"]["predicted"] = False
            break

    for kw, cat in [
        ("call mom", "personal"),
        ("call dad", "personal"),
        ("grocery shopping", "shopping"),
        ("pick up kids", "family"),
        ("practice guitar", "creative"),
        ("workout", "fitness"),
        ("doctor", "health"),
        ("dentist", "health"),
        ("meditat", "health"),
        ("study", "study"),
        ("exam", "study"),
        ("homework", "study"),
        ("rent", "finance"),
        ("tax", "finance"),
        ("bill", "finance"),
        ("invoice", "finance"),
        ("laundry", "home"),
        ("clean", "home"),
        ("gym", "fitness"),
        ("yoga", "fitness"),
        ("standup", "work"),
        ("meeting", "work"),
        ("presentation", "work"),
        ("call", "work"),
        ("code", "work"),
        ("coding", "work"),
        ("flight", "travel"),
        ("hotel", "travel"),
        ("book flight", "travel"),
        ("kids", "family"),
        ("children", "family"),
        ("blog", "creative"),
        ("guitar", "creative"),
        ("spanish", "learning"),
        ("buy", "shopping"),
        ("shopping", "shopping"),
        ("supermarket", "shopping"),
    ]:
        if kw in s:
            schema["category"]["value"] = cat
            schema["category"]["predicted"] = False
            break

    return schema
