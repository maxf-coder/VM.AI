# VM.AI Backend Architecture & Implementation Guide
**Version:** 3.1 (Updated)
**Last Updated:** May 8, 2026
**Competition:** ONIA 2026

---

## 1. Locked Constraints & Competition Rules

| Constraint | Why It Exists | Impact on Architecture |
|------------|---------------|------------------------|
| **v3.0 Database Schema** | State derived from table presence. No `status` field. Draft pattern. | Eliminates state-sync bugs. Requires strict cascade rules. |
| **v3.0 API Contracts** | 15 endpoints, strict validation, `draft_id` for commit flow. | Guarantees frontend-backend parity. |
| **Stable Incremental Scheduler** | Prevents schedule thrashing. | 12s timeout, 1-layer displacement, 25% value threshold. |
| **Synchronous Stats Recorder** | Eliminates race conditions. | Runs in same DB transaction. |
| **Atomic Commits** | Prevents blank calendar. | Single PostgreSQL transaction. |
| **PostgreSQL-Native Types** | Performance & type safety. | UUID, JSONB, DATETIME. |

---

## 2. Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application Layer                │
│  • Routers (endpoints/)                                     │
│  • Dependency Injection (get_db)                         │
│  • CORS Middleware                                        │
│  • Background Tasks (Cleanup Loop)                         │
└───────────────────────┬─────────────────────────────────────┘
                         │ HTTP Request/Response
┌───────────────────────▼─────────────────────────────────────┐
│                    Validation Layer (Pydantic)              │
│  • Request/Response schemas                                │
│  • Strict type checking                                 │
│  • Model validators                                   │
└───────────────────────┬─────────────────────────────────────┘
                         │ Validated Objects
┌───────────────────────▼─────────────────────────────────────┐
│                    Business Logic Layer (services/)         │
│  • NLP Parser → Task Matching → Enrichment → Scheduler   │
└───────────────────────┬─────────────────────────────────────┘
                         │ SQLAlchemy Queries
┌───────────────────────▼─────────────────────────────────────┐
│                    Data Access Layer (models/)              │
└───────────────────────┬─────────────────────────────────────┘
                         │ SQL Execution
┌───────────────────────▼──────────────────────────���──────────┐
│                    PostgreSQL Database                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
src/backend/
├── app/
│   ├── main.py                    # FastAPI app, CORS, routers, lazy loading, health check, cleanup loop
│   ├── core/
│   │   ├── config.py             # Settings
│   │   ├── database.py          # SQLAlchemy engine, session
│   │   └── logging_config.py   # Logging
│   ├── api/v1/endpoints/
│   │   ├── tasks.py            # Task CRUD + NLP parsing + image parsing
│   │   ├── schedule.py         # Schedule fetching + batch
│   │   ├── provisional.py     # Provisional management
│   │   ├── duration.py         # Duration prediction
│   │   └── stats.py            # Task rating
│   ├── models/
│   │   ├── task.py            # tasks table
│   │   ├── schedule.py        # main/provisional_schedule
│   │   ├── workflow.py        # unscheduled_tasks, schedule_changes
│   │   ├── statistics.py      # tasks/category_statistics
│   │   ├── category.py       # categories
│   │   ├── location.py       # locations
│   │   ├── task_category.py  # task_categories junction
│   │   ├── draft.py        # task_drafts
│   │   └── base.py         # Base model
│   ├── schemas/
│   │   ├── task.py          # TaskPayload, responses
│   │   ├── schedule.py    # Schedule responses
│   │   ├── stats.py       # RateRequest/Response
│   │   ├── enrichment.py # TaskPayloadComputed, refs
│   │   ├── nlp.py        # NlpAddPayload, NlpPayloadField
│   │   ├── task_matcher.py # MatchResult
│   │   ├── duration.py    # DurationPredictRequest, DurationPredictResponse
│   │   └── shared.py     # SuccessResponse
│   ├── services/
│   │   ├── task_matcher.py   # MiniLM embeddings
│   │   ├── enrichment.py    # Date resolution, overwrites
│   │   ├── schedule_engine.py # Stable incremental scheduler
│   │   ├── stats_recorder.py # Two-denominator statistics
│   │   ├── parser.py        # T5 NLP parser
│   │   ├── duration.py      # Duration predictor (XGBoost)
│   │   └── img_to_prompt.py # EfficientNet-B4 image classifier
│   ├── utils/
│   │   ├── model_loader.py   # Pre-loads all AI models
│   │   ├── cleanup.py        # Background cleanup loop (drafts, decay)
│   │   ├── task_saver.py     # ORM persistence for tasks
│   │   └── task_normalizer.py# Normalize task payloads
├── alembic/                   # Migrations
├── logs/                     # backend.log
└── pyproject.toml           # Dependencies
```

---

## 4. Key Technologies

### 4.1 FastAPI
- Automatic OpenAPI docs at `/docs`
- Dependency injection for database sessions
- Background async tasks

### 4.2 SQLAlchemy 2.0
- ORM mappings for all tables
- Relationships with cascaded deletes
- JSONB for flexible fields

### 4.3 Pydantic v2
- Strict validation with Field constraints
- Model validators for cross-field logic
- Datetime UUID automatic validation

### 4.4 PostgreSQL 15+
- UUID primary keys
- JSONB for statistics
- Atomic transactions
- FOR UPDATE SKIP LOCKED

---

## 5. The 5-Stage Pipeline

### Stage 1: NLP Parser
- Input: Natural language
- Output: TaskPayload with {value, predicted} structure
- Model: T5-base fine-tuned

### Stage 2: Task Matching
- Input: Task name
- Process: Exact match → MiniLM → Cosine similarity
- Thresholds: ≥0.92 "same", 0.65-0.91 "similar", <0.65 "none"
- Output: {associated_id, association_status, name_vector}

### Stage 3: Enrichment
- Input: TaskPayload + Match result
- Process: Date parse → Overwrite predicted → Compute urgency/value
- Priority: task_statistics (records≥3) → category_statistics → keep value
- Output: Full task data for DB

### Stage 4: Scheduler
- Input: Unscheduled tasks (FIFO)
- Process: Constraint solver → Score all slots (TOP_N_CANDIDATES=400) → Stable scoring
- Constraints: 12s timeout, 1-layer displacement, 25% value threshold
- Output: provisional_schedule + schedule_changes

### Stage 5: Stats Recorder
- Input: Completed task + actual values
- Process: Synchronous update
- Denominators: records (plan), completed_count (delta)
- Output: Updated statistics

---

## 6. API Endpoints (15 Total)

### Tasks Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/tasks/parse/add` | Parse NLP to draft |
| POST | `/tasks/parse/modify` | Parse NLP to modify |
| POST | `/tasks/parse/from-image` | Parse image to task prompt |
| POST | `/tasks/predict-duration` | Predict task duration |
| POST | `/tasks` | Create task |
| POST | `/tasks/{id}/update` | Update task |
| DELETE | `/tasks/{id}` | Delete task |
| GET | `/tasks/{id}` | Get task |
| GET | `/tasks/unscheduled` | Get queue |

### Schedule Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/schedule` | Get main schedule |
| POST | `/schedule/batch` | Run scheduler |

### Provisional Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/provisional/changes` | Get pending |
| POST | `/provisional/reset` | Reset |
| POST | `/provisional/commit` | Commit |

### Stats Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/tasks/{id}/rate` | Rate task |

---

## 7. Database Schema (v3.0)

### Core Tables
- **tasks**: Primary task storage
- **main_schedule**: Committed calendar
- **provisional_schedule**: Working copy
- **schedule_changes**: Change log

### Workflow Tables
- **unscheduled_tasks**: FIFO queue
- **task_drafts**: Temporary drafts

### Statistics Tables (v3.0 - with bucket counts)
- **tasks_statistics**: Task-level with avg_duration{bucket: {count, avg}}
- **category_statistics**: Category-level with bucket counts

### Normalization Tables
- **categories**: Master list
- **locations**: Master list
- **task_categories**: Junction
- **task_statistics_locations**: Location tracking
- **category_statistics_locations**: Location tracking

---

## 8. Data Flow Examples

### Manual Task Creation
```
Frontend → POST /tasks (TaskPayload)
         → Task Matching (MiniLM)
         → Enrichment (compute urgency/value)
         → DB: create task, categories
         → DB: add to unscheduled queue
         → Response: task_id, status
```

### NLP Task Creation
```
Frontend → POST /tasks/parse/add (prompt)
         → NLP Parser (T5)
         → DB: save to task_drafts
         → Response: draft_id, TaskPayload
         
Frontend → POST /tasks (task + draft_id)
         → Load draft → merge → save
```

### NLP Task Modification
```
Frontend → GET /tasks/{id} (get task)
         → POST /tasks/parse/modify (task, prompt)
         → Merge changes
         → Response: modified TaskPayload
```

---

## 9. Field Overwrite Logic (Enrichment)

### Priority Chain
1. task_statistics (if records ≥ 3)
2. category_statistics (loop by priority)
3. Keep predicted value

### Duration Buckets
Buckets: 0.0, 0.5, 1.0
- If difficulty predicted → use stats difficulty for bucket
- If difficulty explicit → use actual difficulty for bucket

### Importance Formula
```
base = nlp_importance
deadline_boost = 0.3 (≤1d) / 0.2 (≤3d) / 0.1 (≤7d) / 0 (otherwise)
completion_boost = completion_rate × 0.2
final = min(1.0, base + deadline_boost + completion_boost)
```

---

## 10. Error Handling

| Status | Meaning | Solution |
|--------|---------|----------|
| 422 | Validation Error | Check request body |
| 404 | Not Found | Verify UUID |
| 409 | Conflict | Check task state |
| 500 | Server Error | Check logs |

---

## 11. Development Commands

```powershell
# Run server
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Generate migration
alembic revision --autogenerate -m "description"
alembic upgrade head

# Run tests
uv run pytest
```

---

*Document prepared for ONIA 2026.*