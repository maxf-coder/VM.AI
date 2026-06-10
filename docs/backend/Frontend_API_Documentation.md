# VM.AI — Frontend API Documentation
**Version:** 3.1 (Current)
**Last Updated:** May 3, 2026
**Competition:** ONIA 2026

---

## 1. API Quick Reference

### Base URL
```
http://localhost:8000/api/v1
```

### DateTime Format (Important)
All datetime values use **naive ISO 8601 format** (no timezone info):
```
"2026-05-09T09:30:00"
```
**Do NOT include** `Z`, `+03:00`, or any timezone suffix.

### All Endpoints (15 Total)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/tasks/parse/add` | Parse NLP to draft |
| POST | `/tasks/parse/modify` | Parse modification prompt |
| POST | `/tasks/parse/from-image` | Parse image to task prompt |
| POST | `/tasks/predict-duration` | Predict task duration |
| POST | `/tasks` | Create task (manual or from draft) |
| POST | `/tasks/{id}/update` | Update existing task |
| DELETE | `/tasks/{id}` | Delete task |
| GET | `/tasks/{id}` | Get task details |
| GET | `/tasks/unscheduled` | Get unscheduled queue |
| GET | `/schedule` | Get main schedule |
| POST | `/schedule/batch` | Run scheduler |
| GET | `/provisional/changes` | Get pending changes |
| POST | `/provisional/reset` | Reset provisional |
| POST | `/provisional/commit` | Commit changes |
| POST | `/tasks/{id}/rate` | Rate task completion |

---

## 2. Request/Response Schemas

### 2.1 SuccessResponse (Base)
```python
class SuccessResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None
```

### 2.2 TaskPayload
```python
class TaskPayload(BaseModel):
    name: str                                    # Must be non-empty
    start: Optional[datetime] = None              # Required if fixed_time=False
    deadline: Optional[datetime] = None           # Required if fixed_time=False
    difficulty: float                         # > 0.0 - 1.0
    duration: int                          # 1 - 1439 minutes
    category: List[str]                    # At least one required
    location: str                             # Required
    importance: float                    # > 0.0 - 1.0
    fixed_time: bool = False
    fixed_start: Optional[datetime] = None   # Required if fixed_time=True
```

---

## 3. Endpoints Detailed

### 3.1 POST /tasks/parse/add — Parse NLP (Add Mode)

**Purpose:** Parse natural language to create a task draft.

**Request:**
```json
{
    "prompt": "finish chemistry homework before Friday"
}
```

**Response (ParseAddResponse):**
```json
{
    "draft_id": "550e8400-e29b-41d4-a716-446655440000",
    "task": {
        "name": "Chemistry homework",
        "start": "2026-05-09T09:00:00",
        "deadline": "2026-05-13T17:00:00",
        "difficulty": 0.7,
        "duration": 60,
        "category": ["study"],
        "location": "Library",
        "importance": 0.6,
        "fixed_time": false,
        "fixed_start": null
    }
}
```

**Schema:** `ParseAddRequest` → `ParseAddResponse`

---

### 3.2 POST /tasks/parse/modify — Parse NLP (Modify Mode)

**Purpose:** Parse natural language to modify an existing task.

**Request:**
```json
{
    "task": {
        "name": "Math homework",
        "start": "2026-05-09T09:00:00",
        "deadline": "2026-05-14T17:00:00",
        "difficulty": 0.7,
        "duration": 60,
        "category": ["study"],
        "location": "Library",
        "importance": 0.6,
        "fixed_time": false,
        "fixed_start": null
    },
    "prompt": "postpone deadline to next week"
}
```

**Response (ParseModifyResponse):**
```json
{
    "task": {
        "name": "Math homework",
        "start": "2026-05-09T09:00:00",
        "deadline": "2026-05-14T17:00:00",
        "difficulty": 0.7,
        "duration": 60,
        "category": ["study"],
        "location": "Library",
        "importance": 0.7,
        "fixed_time": false,
        "fixed_start": null
    }
}
```

**Schema:** `ParseModifyRequest` → `ParseModifyResponse`

---

### 3.3 POST /tasks — Create Task

**Purpose:** Create a task either manually or from a draft.

**Request (TaskCreateRequest):**
```json
{
    "task": {
        "name": "Math homework",
        "start": "2026-05-09T09:00:00",
        "deadline": "2026-05-14T17:00:00",
        "difficulty": 0.7,
        "duration": 60,
        "category": ["study"],
        "location": "Library",
        "importance": 0.6,
        "fixed_time": false,
        "fixed_start": null
    },
    "draft_id": null  // Optional: UUID if committing from NLP
}
```

**Response (TaskResponse):**
```json
{
    "success": true,
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "unscheduled",
    "message": "Task created successfully"
}
```

**Schema:** `TaskCreateRequest` → `TaskResponse` (includes `SuccessResponse`)

**Validation Rules:**
- If `fixed_time=True`: `start` and `deadline` must be null, `fixed_start` required
- If `fixed_time=False`: `start` and `deadline` required, `fixed_start` must be null
- `difficulty`: > 0.0 - 1.0
- `duration`: 1 - 1439
- `importance`: > 0.0 - 1.0

---

### 3.4 POST /tasks/{id}/update — Update Task

**Purpose:** Update an existing task.

**Request (TaskUpdateRequest):**
```json
{
    "task": {
        "name": "Updated Math homework",
        "start": "2026-05-10T10:00:00",
        "deadline": "2026-05-15T17:00:00",
        "difficulty": 0.8,
        "duration": 90,
        "category": ["study"],
        "location": "Library",
        "importance": 0.7,
        "fixed_time": false,
        "fixed_start": null
    }
}
```

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| source | string | Yes | `main_schedule` \| `unscheduled` \| `provisional` |

**Validation Rules (main_schedule only):**
- Cannot update task if `slot.end < now` (task ended in the past) → 400
- Cannot update task if `task.rated == True` → 409

**Response (TaskResponse):**
```json
{
    "success": true,
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "unscheduled",
    "message": "Task updated successfully"
}
```

**Schema:** `TaskUpdateRequest` → `TaskResponse`

---

### 3.5 DELETE /tasks/{id} — Delete Task

**Purpose:** Delete a task from the system.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| source | string | Yes | `main_schedule` \| `unscheduled` \| `provisional` \| `tasks` |

**Source Options:**
- `main_schedule`: Verifies task is in main_schedule, then deletes from tasks table
- `unscheduled`: Removes from unscheduled queue or tasks table
- `provisional`: Removes from provisional schedule
- `tasks`: Directly deletes from tasks table

**Response:** 204 No Content (empty)

**Schema:** Returns nothing on success

---

### 3.6 GET /tasks/{id} — Get Task Details

**Purpose:** Retrieve a specific task by ID.

**Response (TaskDetailResponse):**
```json
{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "task": {
        "name": "Math homework",
        "start": "2026-05-09T09:00:00",
        "deadline": "2026-05-14T17:00:00",
        "difficulty": 0.7,
        "duration": 60,
        "category": ["study"],
        "location": "Library",
        "importance": 0.6,
        "fixed_time": false,
        "fixed_start": null
    },
    "created_at": "2026-05-15T10:30:00"
}
```

**Schema:** `TaskDetailResponse`

**Error:** 404 if task not found

---

### 3.7 GET /tasks/unscheduled — Get Unscheduled Queue

**Purpose:** Get list of tasks waiting for scheduling.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| limit | int | No | Max tasks to return (default: none, returns all) |

**Response (UnscheduledResponse):**
```json
{
    "tasks": [
        {
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "task": {
                "name": "Math homework",
                "start": "2026-05-09T09:00:00",
                "deadline": "2026-05-14T17:00:00",
                "difficulty": 0.7,
                "duration": 60,
                "category": ["study"],
                "location": "Library",
                "importance": 0.6,
                "fixed_time": false,
                "fixed_start": null
            },
            "created_at": "2026-05-15T10:30:00"
        }
    ],
    "total_count": 1
}
```

**Schema:** `UnscheduledResponse`

---

### 3.8 GET /schedule — Get Main Schedule

**Purpose:** Retrieve committed schedule for a specific date.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| date | date | Yes | Date in YYYY-MM-DD format |

**Response (ScheduleResponse):**
```json
{
    "date": "2026-05-20",
    "tasks": [
        {
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "Math homework",
            "start": "2026-05-20T09:00:00",
            "end": "2026-05-20T10:30:00",
            "location": "Library",
            "rated": false
        }
    ]
}
```

**Schema:** Returns `ScheduleResponse` with date and list of `ScheduleTask`

---

### 3.9 POST /tasks/{id}/rate — Rate Task

**Purpose:** Rate a completed or uncompleted task.

**Request Body (RateRequest):**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| completed | bool | Yes | Whether the task was completed |
| actual_duration | int | If completed | Actual duration in minutes (1-1439) |
| actual_difficulty | float | If completed | Actual difficulty (0.0-1.0) |

**Example:**
```json
{
    "completed": true,
    "actual_duration": 45,
    "actual_difficulty": 0.5
}
```

**Response (RateResponse):**
```json
{
    "success": true,
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Task rated successfully"
}
```

**Validation Rules:**
| Check | HTTP | Error |
|-------|------|-------|
| Task in main_schedule | 400 | "Task not in main schedule" |
| Task.rated == True | 400 | "Task already rated" |

**Schema:** `RateResponse`

---

### 3.10 POST /schedule/batch — Run Scheduler

**Purpose:** Run the scheduling algorithm on unscheduled tasks.

**Response (BatchScheduleResponse):**
```json
{
    "success": true,
    "scheduled_count": 5,
    "failed_count": 0,
    "unscheduled_remaining": [],
    "results": [
        {
            "success": true,
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "slot_id": "660e8400-e29b-41d4-a716-446655440001",
            "slot_start": "2026-05-20T09:00:00",
            "slot_end": "2026-05-20T10:30:00",
            "displaced_tasks": [],
            "message": "Task scheduled successfully"
        }
    ],
    "execution_time_ms": 3500
}
```

**Note:** Service layer returns `BatchSchedulingResult`, endpoint converts to `BatchScheduleResponse`.

**Schema:** `BatchScheduleResponse`

---

### 3.11 GET /provisional/changes — Get Pending Changes

**Purpose:** Get scheduled changes before committing.

**Response (ProvisionalChangesResponse):**
```json
{
    "changes": [
        {
            "provisional_schedule_slot_id": "660e8400-e29b-41d4-a716-446655440001",
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "task_name": "Math homework",
            "change_type": "insert",
            "new_slot_start": "2026-05-20T09:00:00",
            "new_slot_end": "2026-05-20T10:30:00",
            "location": "Library"
        }
    ],
    "total_count": 1
}
```

**Schema:** `ProvisionalChangesResponse`

---

### 3.12 POST /provisional/reset — Reset Provisional

**Purpose:** Discard provisional schedule and reset to main schedule.

**Response (ProvisionalResetResponse):**
```json
{
    "success": true,
    "message": "Provisional schedule reset to main schedule",
    "changes_discarded": 3
}
```

**Schema:** `ProvisionalResetResponse`

---

### 3.13 POST /provisional/commit — Commit Changes

**Purpose:** Atomically copy provisional to main schedule.

**Response (ProvisionalCommitResponse):**
```json
{
    "success": true,
    "committed_count": 3,
    "message": "Schedule committed successfully",
    "transaction_time_ms": 15
}
```

**Schema:** `ProvisionalCommitResponse`

---

### 3.14 POST /tasks/parse/from-image — Parse Image to Task Prompt

**Purpose:** Upload an image and classify the activity using EfficientNet-B4, returning a task prompt string.

**Request:**
```json
{
    "image": "<base64-encoded image data>"
}
```

**Response (ImageParseResponse):**
```json
{
    "success": true,
    "prompt": "Finish chemistry homework"
}
```

**Schema:** `ImageParseRequest` → `ImageParseResponse`

---

### 3.15 POST /tasks/predict-duration — Predict Task Duration

**Purpose:** Predict task duration from task attributes using the XGBoost duration predictor.

**Request:**
```json
{
    "difficulty": 0.7,
    "importance": 0.6,
    "scheduled_duration": 90,
    "category": "study",
    "location": "Library",
    "fixed_time": false,
    "time_difference": 120
}
```

**Response (DurationPredictResponse):**
```json
{
    "predicted_duration": 60
}
```

**Schema:** `DurationPredictRequest` → `DurationPredictResponse`

**Validation:**
- `difficulty`: > 0.0 - 1.0
- `importance`: > 0.0 - 1.0
- `scheduled_duration`: 1 - 1439
- `time_difference`: minutes until deadline (non-negative)

---

## 4. Error Handling

| Status Code | Meaning | Cause | Solution |
|------------|---------|-------|-----------|
| 422 | Validation Error | Missing field, invalid range, bad datetime format | Check request body |
| 404 | Not Found | Invalid UUID or missing task | Verify task_id exists |
| 409 | Conflict | Task already rated, duplicate schedule | Check task state |
| 500 | Server Error | Catch-all | Check backend logs |

---

*Document prepared for ONIA 2026.*