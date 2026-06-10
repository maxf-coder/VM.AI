# VM.AI — Full Project Overview
**Competition:** ONIA 2026
**Status:** Architecture Complete, Full Pipeline Implemented

---

## 1. Core Value Proposition

VM.AI is an AI-driven personal scheduling system that transforms natural language input into optimized, behavior-aware calendar schedules. It prioritizes **schedule stability** and **predictable performance**.

The system learns from user's task completion patterns to make intelligent predictions about task duration, difficulty, and importance, while ensuring scheduled tasks don't thrash the user's calendar unexpectedly.

---

## 2. The 7-Stage Pipeline

### Stage 1: NLP Parser
- **Input:** Natural language text (e.g., "finish chemistry homework before Friday")
- **Process:** T5-base conditional generation → JSON parsing → Pydantic validation
- **Output:** `TaskPayload` with `{value, predicted}` structure for each field
- **Model:** Fine-tuned T5-base transformer
- **Workflow:** Saves to `task_drafts` table → Returns `draft_id` to frontend

### Stage 2: Task Matching
- **Input:** Parsed task name
- **Process:** 
  1. Exact case-insensitive match
  2. MiniLM embeddings + Cosine similarity
  3. Threshold classification
- **Thresholds:** 
  - ≥0.92 → `"same"` (identical task)
  - 0.65-0.91 → `"similar"` (related task)
  - <0.65 → `"none"` (new task)
- **Output:** `{associated_id, association_status, name_vector}`
- **Invariant:** `associated_id` points to `tasks_statistics.id`, never `tasks.id`

### Stage 3: Enrichment
- **Input:** Parsed task + Match result
- **Process:** 
  1. Date resolution (dateparser strict config)
  2. Historical averaging based on match status
  3. Importance recomputation with deadline boost
  4. Compute urgency and value
- **Priority Chain:**
  - Task statistics (only if `records >= 3`)
  - Category statistics (loop by priority)
  - Keep predicted value (cold start defaults to 0.5)
- **Output:** Full task data ready for DB insertion

### Stage 4: Scheduler (Complete - v4.0)
- **Algorithm:** Stable Incremental Algorithm
- **Implementation:** ScheduleEngine class
- **Process:**
  1. Fetch tasks (hybrid: queue or task_ids list)
  2. Build free slot inventory
  3. Constraint solver
  4. **ALL slots scored** (not just top-15)
  5. Stable scoring with displacement handling
- **Scoring Formula:**
  ```
  score = BASE_SLOT_SCORE + location_boost + free_slot_boost + time_score_boost + urgency_boost + continuity_boost - overlap_penalty
  ```
- **Key Parameters:**
  - BASE_SLOT_SCORE=1.0 (baseline for all slots)
  - TOP_N_CANDIDATES=400 (score all candidate slots)
  - FREE_SLOT_BOOST=0.5 (free slots bubble to top)
  - MAX_LAYER=1 (1-layer displacement max)
  - VALUE_THRESHOLD=1.25 (new task must be 25% more valuable to displace)
- **Output:** Writes to `provisional_schedule` + `schedule_changes`

### Stage 5: Stats Recorder
- **Execution:** Synchronous in same DB transaction
- **Two Denominators:**
  - Plan averages → `records` (updated on commit)
  - Delta averages → `completed_count` (updated on rating)
- **Time Score Decay:**
  - All time scores multiplied by `×0.99` every 24 hours via background cleanup loop
  - Values below `0.1` threshold are zeroed out
  - Prevents stale preference scores from accumulating

### Stage 6: Image-to-Prompt
- **Input:** Base64-encoded image
- **Process:** EfficientNet-B4 image classification → activity detection → prompt generation
- **Output:** Task prompt string (e.g., "Finish chemistry homework")
- **Endpoint:** `POST /tasks/parse/from-image`

### Stage 7: Duration Prediction
- **Input:** Task attributes (difficulty, importance, scheduled_duration, category, location, fixed_time, time_difference)
- **Process:** XGBoost regressor prediction
- **Output:** Predicted duration in minutes
- **Endpoint:** `POST /tasks/predict-duration`

---

## 3. Key Features

### 3.1 Class-Based Service Architecture
All core services follow a class-based pattern:
- **EnrichmentService** - Field enrichment, priority chain, importance recomputation
- **ScheduleEngine** - Scheduling with ALL slots scoring, displacement handling
- **TaskMatchingService** - Embedding-based task association
- **StatsRecorderService** - Two-denominator statistics updates
- **DurationService** - XGBoost-based duration prediction
- **ImgToPrompt** - EfficientNet-B4 image classification
- **Schedule batch** - Schedules all tasks from the unscheduled queue (FIFO) with time score updates

### 3.2 Draft System
- **Purpose:** Safe task creation via Chat/AI without polluting the main database
- **Flow:**
  1. User enters NLP prompt
  2. NLP Parser saves to `task_drafts`
  3. Frontend receives `draft_id` + clean `TaskPayload`
  4. User can edit fields before commit
  5. On commit: draft loaded → merged → saved to `tasks`
- **Safety:** If user abandons, background cleanup deletes drafts after 24 hours

### 3.3 Strict Validation
- All API schemas use Pydantic `Field` with constraints
- Automatic datetime ISO validation
- Model validators for `fixed_time` logic
- Catches 80% of errors at the boundary

### 3.4 Behavior-Aware Predictions
- Task-level statistics (from matched tasks)
- Category-level aggregates (fallback)
- Importance recalculation with deadline proximity
- Location preferences

### 3.5 Stable Scheduling
- Incremental updates (not full reschedule)
- 1-layer displacement maximum
- 25% value threshold
- **ALL slots scored** (not top-15) - ensures low-value tasks find free slots

### 3.6 Atomic Commits
- Single PostgreSQL transaction for schedule commit
- Prevents blank calendar on network drop

---

## 4. User Workflows

### 4.1 Manual Task Creation
```
1. User fills all fields in frontend form
2. Frontend calls POST /tasks with TaskPayload
3. Backend calls Task Matching
4. Backend computes urgency/value (EnrichmentService)
5. Backend creates task + statistics + categories
6. Backend adds to unscheduled queue
7. Return task_id + status
```

### 4.2 NLP Task Creation (Add Mode)
```
1. User enters NLP prompt
2. Frontend calls POST /tasks/parse/add
3. Backend runs NLP Parser → TaskPayload
4. Backend saves to task_drafts → returns draft_id
5. Frontend shows preview + edit options
6. User edits (optional)
7. Frontend calls POST /tasks with draft_id
8. Backend loads draft → merges with edits
9. Continue as Manual Creation
```

### 4.3 NLP Task Modification
```
1. User selects task to modify
2. Frontend calls GET /tasks/{id}
3. User enters modification prompt
4. Frontend calls POST /tasks/parse/modify (task: TaskPayload, prompt)
5. Backend runs NLP Parser on changes
6. Backend merges changes with existing
7. Frontend shows modified task
```

### 4.4 Schedule Batch (Queue-based)
```
1. Frontend calls POST /schedule/batch (no parameters)
2. Backend fetches all tasks from unscheduled_queue (FIFO order)
3. ScheduleEngine.schedule_batch():
   a. Build free slot inventory
   b. For each task:
      - Get candidate slots (time window constraint solving)
      - Score ALL slots (TOP_N_CANDIDATES=400)
      - Place in highest-scoring slot
      - Handle displacement if needed (MAX_LAYER=1)
      - On success: update time score (+1.0 boost)
4. Save to provisional_schedule
5. Record schedule_changes
```

---

## 5. API Endpoints Summary

### Tasks
| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST | `/tasks` | `TaskCreateRequest` | `TaskResponse` |
| GET | `/tasks/{id}` | - | `TaskDetailResponse` |
| POST | `/tasks/{id}/update` | `TaskUpdateRequest` | `TaskResponse` |
| DELETE | `/tasks/{id}` | - | (204 No Content) |
| GET | `/tasks/unscheduled` | - | `UnscheduledResponse` |
| POST | `/tasks/parse/add` | `ParseAddRequest` | `ParseAddResponse` |
| POST | `/tasks/parse/modify` | `ParseModifyRequest` | `ParseModifyResponse` |
| POST | `/tasks/parse/from-image` | `ImageParseRequest` | `ImageParseResponse` |
| POST | `/tasks/predict-duration` | `DurationPredictRequest` | `DurationPredictResponse` |

### Schedule
| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| GET | `/schedule` | - | `ScheduleResponse` |
| POST | `/schedule/batch` | - | `BatchScheduleResponse` |

### Provisional
| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| GET | `/provisional/changes` | - | `ProvisionalChangesResponse` |
| POST | `/provisional/commit` | - | `SuccessResponse` |
| POST | `/provisional/reset` | - | `SuccessResponse` |

### Stats
| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST | `/tasks/{id}/rate` | `RateRequest` | `RateResponse` |

---

## 6. Error Handling

| Status Code | Meaning | Cause |
|------------|---------|-------|
| 422 | Validation Error | Pydantic rejects malformed input |
| 404 | Not Found | Invalid UUID or missing task |
| 409 | Conflict | Task already rated, duplicate scheduled |
| 500 | Server Error | Catch-all, log boundary, return safe message |

---

## 7. Technology Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI |
| ORM | SQLAlchemy 2.0 |
| Database | PostgreSQL 15+ |
| Migrations | Alembic |
| Validation | Pydantic v2 |
| NLP Model | T5-base (fine-tuned) |
| Image Classifier | EfficientNet-B4 |
| Duration Predictor | XGBoost regressor |
| Embeddings | MiniLM |
| Date Parsing | dateparser |

---

## 8. Development Status

| Component | Status |
|-----------|--------|
| API Schemas | Complete |
| Database Models | Complete |
| Task Matching | Complete |
| NLP Parser | Complete |
| Image Classifier | Complete (ImgToPrompt, EfficientNet-B4) |
| Enrichment | Complete (EnrichmentService class) |
| Scheduler | Complete (ScheduleEngine, ALL slots scored) |
| Duration Predictor | Complete (DurationService, XGBoost) |
| Stats Recorder | Complete (StatsRecorderService class) |

---

## 9. Key Implementation Details

### 9.1 ScheduleEngine Scoring
```
score = BASE_SLOT_SCORE + location_boost + free_slot_boost + time_score_boost + urgency_boost + continuity_boost - overlap_penalty
```

Where:
- **BASE_SLOT_SCORE** = 1.0 (baseline for all slots)
- **location_boost** = 0.0 – 0.25 (LOCATION_BASE_BOOST × continuity_count; max when same location both before and after)
- **free_slot_boost** = 0.5 (FREE_SLOT_BOOST) if slot has no overlapping tasks, else 0
- **time_score_boost** = -0.3 – +0.3 (TIME_SCORE_AMPLIFIER × score/10; from task or category time preferences)
- **urgency_boost** = 0 – 0.21 (URGENCY_AMPLIFIER × urgency_value × (1 - position_ratio); higher for slots closer to now)
- **continuity_boost** = 0, 0.05, or 0.1 (CONTINUITY_BASE_BOOST based on gap: 0.05 for 15min, 0.1 for 30min, 0.05 for 45min)
- **overlap_penalty** = 0.15 per overlapping task (OVERLAP_BASE_PENALTY × overlap_count)

### 9.2 Displacement Handling
- Only if `new_task.value > existing_task.value × VALUE_THRESHOLD (1.25)` — new task must be 25% more valuable
- MAX_LAYER=1 prevents cascade rescheduling
- Displaced tasks are rescheduled via `_try_reschedule_task()` at layer+1; if rescheduling fails, the original task is not placed

### 9.3 Queue-Based Batch Schedule
```python
def schedule_batch(self, db: Session) -> BatchSchedulingResult:
    tasks = db.query(UnscheduledTask).order_by(UnscheduledTask.created_at).all()
    for entry in tasks:
        result = self.schedule_single(entry.task, db)
        if result.success:
            stats_recorder.update_time_score(db, entry.task.id, result.slot_start, boost=1.0)
```

---

---

## 10. Ethics and Impact

### 10.1 Privacy & Data Storage
- Task data is stored in PostgreSQL (single-user demo, no user_id fields)
- Data is NOT encrypted at rest — known demo limitation
- No telemetry, analytics, tracking, or third-party data sharing
- No personal or confidential data is included in the repository

### 10.2 Bias & Fairness
- The T5 parser was fine-tuned on English template data only
- Performance may degrade for non-English input, slang, dialect, or creative phrasing
- The XGBoost regressor was trained on ~1000 labeled examples from a single user — predictions reflect that user's labeling patterns and may not generalize
- No systematic bias analysis has been performed — this is a known limitation

### 10.3 Known Risks
- Duration predictions have MAE ≈ 10 minutes — do not rely on them for critical scheduling
- The scheduler has known limitations with overnight tasks (naive datetime, no timezone)
- All ML predictions are estimates; users should verify before committing
- The system is a demo/prototype, not a production scheduling tool

### 10.4 Responsible Use
- Always review scheduled tasks before accepting automated changes
- Report unexpected behavior via GitHub Issues
- This is an assistive tool — final scheduling decisions remain with the user

### 10.5 Transparency
- Known bugs are documented in `src/backend/logs/`
- Model limitations are discussed in this section
- No deliberate manipulation of results

---

*Document prepared for ONIA 2026.*