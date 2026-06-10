# VM.AI — Enrichment Module Documentation
**Version:** 1.0 (Final)
**Last Updated:** April 18, 2026
**Competition:** ONIA 2026

---

## 1. Overview

The Enrichment module is responsible for transforming raw task data (either from NLP or user input) into complete, database-ready records. It handles:

1. **Date Resolution:** Converting natural language dates to datetime objects
2. **Field Overwriting:** Applying historical data to predicted fields
3. **Value Computation:** Calculating urgency and composite task value
4. **Draft Management:** Saving/loading temporary drafts

---

## 2. Architecture

### 2.1 Two-Phase Execution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENRICHMENT SERVICE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Predict (NLP add mode)                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  1. Parse dates (dateparser)                                         │ │
│  │  2. Build overwrite map (task_stats → category_stats → keep)         │ │
│  │  3. Overwrite predicted fields (difficulty, duration, importance)  │ │
│  │  4. Save to draft table                                              │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                          │
│  Phase 2: Commit                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  1. Load draft (if from draft)                                        │ │
│  │  2. Merge request with draft (request priority)                       │ │
│  │  3. Compute urgency/value                                           │ │
│  │  4. Add internal references                                        │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Public API

| Method | Input | Output | Used In |
|--------|-------|--------|--------|
| `predict_nlp_add(db, nlp_payload, match_result)` | `{field: {value, predicted}}` | `(TaskPayload, draft_id)` | NLP add flow |
| `commit_from_draft(db, request_task, draft_id)` | `{field: value}` + draft_id | `full_task_data` | Draft commit |
| `commit_manual(db, task_payload, match_result)` | `TaskPayload` | `full_task_data` | Manual creation |
| `merge_nlp_modify(db, existing_task, changed_fields)` | existing + changes | `merged_task` | NLP modify |
| `update_task(db, task_payload)` | `TaskPayload` | `TaskPayload with computed` | Update from schedule |

---

## 3. Input Structures

### 3.1 NLP Payload (predict_nlp_add)

The NLP parser outputs tasks with a `{value, predicted}` structure indicating whether the value was predicted by the AI or explicitly extracted:

```python
nlp_payload = {
    "name": {"value": "Math homework", "predicted": False},
    "start": {"value": "2026-04-20T09:00:00", "predicted": True},
    "deadline": {"value": "2026-04-25T17:00:00", "predicted": True},
    "difficulty": {"value": 0.7, "predicted": True},
    "duration": {"value": 60, "predicted": True},
    "category": {"value": ["study"], "predicted": False},
    "location": {"value": "Library", "predicted": True},
    "importance": {"value": 0.6, "predicted": True},
}
```

### 3.2 Request Task (commit_from_draft)

Frontend sends clean TaskPayload (no predicted flags - user edits are source of truth):

```python
request_task = {
    "name": "Math homework",
    "start": datetime(2026, 4, 20, 9, 0),
    "deadline": datetime(2026, 4, 25, 17, 0),
    "difficulty": 0.7,
    "duration": 60,
    "category": ["study"],
    "location": "Library",
    "importance": 0.6,
    "fixed_time": False,
    "fixed_start": None,
}
```

### 3.3 Task Payload (commit_manual)

For manual creation, all fields are explicit (user-provided):

```python
task_payload = TaskPayload(
    name="Math homework",
    start=datetime(2026, 4, 20, 9, 0),
    deadline=datetime(2026, 4, 25, 17, 0),
    difficulty=0.7,
    duration=60,
    category=["study"],
    location="Library",
    importance=0.6,
    fixed_time=False,
    fixed_start=None,
)
```

---

## 4. Field Overwrite Logic

### 4.1 Overwrite Decision Chain

For each field where `predicted: true`, the system checks:

```
1. If task_statistics exists AND records >= 3:
   → Use task_statistics data
2. Loop through categories (by priority):
   → Use first category with data for that field
3. If nothing found:
   → Keep predicted value (log warning)
```

### 4.2 Difficulty Overwrite

```python
# Formula: avg_difficulty + avg_difficulty_delta
overwrite_value = stats.avg_difficulty + stats.avg_difficulty_delta
```

### 4.3 Duration Overwrite with Bucket Logic

Duration depends on difficulty. Buckets are: `0.0`, `0.5`, `1.0`

**Bucket Calculation:**
```python
bucket = round(difficulty * 2) / 2  # 0.7 → 0.5, 0.9 → 1.0
```

**Duration Source Priority:**
1. If `difficulty` is predicted → use statistics difficulty for bucket lookup
2. If `difficulty` is explicit → use actual difficulty for bucket lookup
3. If bucket not found → loop through all buckets in statistics
4. If still not found → continue to next category

**Duration Structure (v3.0):**
```python
avg_duration = {
    "0.0": {"count": 5, "avg": 30},
    "0.5": {"count": 3, "avg": 45},
    "1.0": {"count": 4, "avg": 45}
}
# Access: duration_map[bucket]["avg"]
```

### 4.4 Importance Recomputation

Importance is recalculated based on deadline proximity and completion rate:

```python
# Base from NLP
base = nlp_importance

# Deadline boost based on days until deadline
deadline_boost = 0.3  if days_left <= 1
               0.2  if days_left <= 3
               0.1  if days_left <= 7
               0.0  otherwise

# Completion rate from statistics
completion_rate = completed_count / (completed_count + uncompleted_count)
# - From matched task if total >= 3
# - From category statistics otherwise
# - Default 0.5 if no data

completion_boost = completion_rate * 0.2

# Final importance
final_importance = min(1.0, base + deadline_boost + completion_boost)
```

### 4.5 Location Overwrite

Location is selected from statistics junction tables:

**Priority:**
1. If `associated_id` exists: Check `task_statistics_locations`
2. If not found: Loop through categories → `category_statistics_locations`
3. Select location with highest `count`

---

## 5. Value Computation

### 5.1 Urgency Calculation

```python
def _calculate_urgency(importance: float, deadline: Optional[datetime]) -> float:
    if not deadline:
        return 0.0
    
    days_left = (deadline - now).total_seconds() / 86400
    
    if days_left <= 0:
        days_left = 0.001
    
    urgency = min(1.0, importance * (1 / days_left) * 3)
    return round(max(0.0, urgency), 4)
```

### 5.2 Value Calculation

```python
def _calculate_value(
    importance: float,
    urgency: float,
    difficulty: float,
    completion_rate: float = 1.0,
) -> float:
    raw_value = (importance * 0.4) + (urgency * 0.4) + (difficulty * 0.2)
    return round(raw_value * completion_rate, 4)
```

---

## 6. Method Implementations

### 6.1 predict_nlp_add()

```python
def predict_nlp_add(
    self,
    db: Session,
    nlp_payload: dict[str, dict[str, Any]],
    match_result: dict[str, Any],
) -> Tuple[dict[str, Any], UUID]:
    """
    NLP add flow (Phase 1 only).
    
    Steps:
        1. Flatten nlp_payload and parse dates first
        2. Rebuild with parsed datetime for importance calculation
        3. Build overwrite map from statistics
        4. Overwrite predicted fields
        5. Save to draft table
    """
    # Step 1: Flatten and parse dates
    flat_payload = {}
    for field, entry in nlp_payload.items():
        value, _ = self._extract_field(entry)
        flat_payload[field] = value
    
    parsed_task = self._date_parse(flat_payload)
    
    # Step 2: Rebuild with datetime for importance
    nlp_payload_with_dates = nlp_payload.copy()
    for field in ["start", "deadline", "fixed_start"]:
        if field in parsed_task and parsed_task[field] is not None:
            nlp_payload_with_dates[field] = {
                "value": parsed_task[field],
                "predicted": nlp_payload.get(field, {}).get("predicted", False),
            }
    
    # Step 3: Get overwrite map
    overwrite_map = self._get_overwrite_map(db, match_result, nlp_payload_with_dates)
    
    # Step 4: Overwrite fields
    enriched_task = self._overwrite_fields(parsed_task, overwrite_map)
    
    # Step 5: Save to draft
    draft_id = self._draft_save(db, enriched_task, match_result)
    
    return output_schema, draft_id
```

### 6.2 commit_from_draft()

```python
def commit_from_draft(
    self,
    db: Session,
    request_task: dict[str, Any],
    draft_id: UUID,
) -> dict[str, Any]:
    """
    Draft commit flow (Phase 2 only).
    
    Steps:
        1. Load draft (including match_result) from DB
        2. Merge request with draft (request priority)
        3. Compute urgency/value
        4. Add internal refs
    """
    # Load draft
    draft_data = self._draft_load(db, draft_id)
    
    # Get match_result from draft
    match_result = draft_data.get("match_result", {})
    
    # Merge request with draft
    merged_task = self._draft_merge(request_task, draft_data)
    
    # Compute
    full_task_data = self._compute(merged_task)
    
    # Add internal refs
    full_task_data = self._add_internal_refs(full_task_data, match_result)
    
    return full_task_data
```

### 6.3 commit_manual()

```python
def commit_manual(
    self,
    db: Session,
    task_payload: dict[str, Any],
    match_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Manual creation flow (Phase 1 + 2 combined).
    
    Since all fields are explicit (no NLP), skip overwrite.
    Compute urgency/value and add internal refs.
    """
    # Compute urgency/value
    full_task_data = self._compute(task_payload)
    
    # Add internal refs
    full_task_data = self._add_internal_refs(full_task_data, match_result)
    
    return full_task_data
```

---

## 7. Internal Helpers

### 7.1 _get_overwrite_map()

Determines which fields to overwrite based on predicted flags and match status:

```python
def _get_overwrite_map(
    self,
    db: Session,
    match_result: dict[str, Any],
    nlp_payload: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Priority chain:
        1. task_statistics (only if records >= 3)
        2. category_statistics (loop through categories by priority)
        3. Keep predicted value
    """
    # Check records >= 3 before using task_statistics
    task_stats = self._get_task_stats(db, stats_id)
    if task_stats and task_stats.get("records", 0) >= 3:
        # Use task_statistics
        ...
    
    # Fall back to category_statistics
    ...
```

### 7.2 _get_value_from_task_stats()

Gets difficulty or duration from task statistics:

```python
def _get_value_from_task_stats(
    self,
    task_stats: dict,
    field: str,
    difficulty: Optional[float] = None,
) -> Optional[float]:
    """
    For difficulty: avg_difficulty + avg_difficulty_delta
    For duration: uses difficulty bucket to lookup avg_duration[bucket]["avg"]
    """
    if field == "difficulty":
        return stats.avg_difficulty + stats.avg_difficulty_delta
    
    elif field == "duration":
        bucket = self._calculate_bucket(difficulty)
        
        # Support both new and old format
        if bucket in duration_map:
            if isinstance(duration_map[bucket], dict):
                return duration_map[bucket].get("avg")
            else:
                # Old format
                return duration_map[bucket]
```

### 7.3 _calculate_importance()

Recalculates importance based on deadline and completion rate:

```python
def _calculate_importance(
    self,
    db: Session,
    base_importance: float,
    deadline: Optional[datetime],
    match_result: dict[str, Any],
) -> float:
    """
    Formula:
        base = nlp_importance
        deadline_boost = 0.3/0.2/0.1/0 based on days_left
        completion_boost = completion_rate * 0.2
        final = min(1.0, base + deadline_boost + completion_boost)
    """
```

---

## 8. Draft Management

### 8.1 Draft Save

```python
def _draft_save(
    self,
    db: Session,
    task_payload: dict[str, Any],
    match_result: dict[str, Any],
) -> UUID:
    content = {
        "task": task_payload,
        "match_result": {
            "associated_id": str(match_result.get("associated_id")),
            "association_status": match_result.get("association_status"),
            "name_vector": match_result.get("name_vector"),
        },
    }
    
    draft = TaskDraft(id=uuid4(), content=content)
    db.add(draft)
    db.commit()
    
    return draft_id
```

### 8.2 Draft Load

```python
def _draft_load(self, db: Session, draft_id: UUID) -> Optional[dict[str, Any]]:
    draft = db.query(TaskDraft).filter(TaskDraft.id == draft_id).first()
    
    if draft:
        # IMPORTANT: Delete draft after loading (memory efficiency)
        content = draft.content
        db.delete(draft)
        db.commit()
        return content
    
    return None
```

---

## 9. Complete Flow Examples

### 9.1 NLP Add Flow

```
Input:
  prompt: "finish chemistry homework before Friday"
  nlp_payload with {value, predicted} structure
  match_result: {association_status: "same", associated_id: UUID}

Process:
  1. Parse dates: "Friday" → datetime(2026, 4, 25, 17, 0)
  2. Get overwrite map:
     - difficulty predicted=True → task_stats.difficulty (records>=3)
     - duration predicted=True → task_stats with bucket 0.5
     - importance predicted=True → recompute with deadline
     - location predicted=True → task_stats_locations
  3. Overwrite predicted fields
  4. Save to draft → return draft_id

Output:
  (TaskPayload, draft_id)
```

### 9.2 Draft Commit Flow

```
Input:
  request_task: {name: "Math", difficulty: 0.7, ...}  # User edits
  draft_id: UUID

Process:
  1. Load draft (contains NLP output + match_result)
  2. Merge: request_task overwrites draft (user edits priority)
  3. Compute: urgency, value
  4. Add internal refs: task_statistics_id, name_vector, status

Output:
  full_task_data ready for DB insertion
```

### 9.3 Manual Creation Flow

```
Input:
  task_payload: TaskPayload with all explicit fields
  match_result: {association_status, associated_id, name_vector}

Process:
  1. No overwrite (all fields explicit)
  2. Compute: urgency, value
  3. Add internal refs

Output:
  full_task_data ready for DB insertion
```

---

## 10. Backward Compatibility

The enrichment service supports both old and new duration formats:

```python
# New format (v3.0)
duration_map = {"0.5": {"count": 3, "avg": 45}}

# Old format (v2.0)
duration_map = {"0.5": 45}

# Both are handled:
if isinstance(duration_map[bucket], dict):
    avg_val = duration_map[bucket].get("avg")
else:
    # Old format
    avg_val = duration_map[bucket]
```

---

*Document prepared for ONIA 2026.*