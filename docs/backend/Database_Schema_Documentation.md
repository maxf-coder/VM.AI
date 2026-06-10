# VM.AI — Database Schema Documentation
**Version:** 3.1 (Current)
**Last Updated:** May 3, 2026
**Competition:** ONIA 2026

---

## 1. Overview

The VM.AI database consists of five logical groups: **Core Tables** (task storage & scheduling), **Workflow Tables** (task queue & change tracking), **Statistics Tables** (behavioral learning data), **Normalization Tables** (categories & locations), and **Draft Tables** (temporary task storage).

### Key Constraints
- **Single-user system** — no `user_id` fields anywhere
- **No status field** — task state is derived from presence in `unscheduled_tasks`, `provisional_schedule`, or `main_schedule`
- **Draft Pattern** — Uses `task_drafts` table to store "pending" tasks from NLP before commit
- **Statistics persistence** — `tasks_statistics` rows are NEVER cascade-deleted when a task is deleted
- **10-day rolling storage window** — scheduled tasks are kept for 3 past days, current day, and 6 future days
- **Naive datetime format** — All datetime columns use `DateTime(timezone=False)` for consistency with frontend

---

## 2. Design Principles

| Principle | Implementation |
|-----------|----------------|
| **One statistics row per task** | Created at task creation time. May be shared if `association_status = "same"`. |
| **Two-link statistics design** | `tasks.task_statistics_id` → this task's own stats. `tasks.associated_task_statistics_id` → matched task's stats (nullable). |
| **Strict cascade boundaries** | Core tables cascade to each other. Statistics tables have `ON DELETE NO ACTION`. |
| **Atomic workflow operations** | Schedule commit, task creation, and stats updates run inside explicit transactions. |
| **Immutable executed tasks** | Tasks with `end < NOW()` or outside the 10-day window cannot be modified or rescheduled. |
| **Draft Cleanup** | Background async task runs every 24 hours to delete old drafts from `task_drafts` table. |
| **Naive Datetime** | All datetime fields use `TIMESTAMP WITHOUT TIME ZONE` - no timezone conversion. |

---

## 3. Table Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORE TABLES                                    │
├─────────────────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│           tasks              │  │  main_schedule  │  │  provisional_schedule│
│                             │  │                 │  │                     │
│ • id (PK, UUID)             │◄─│  • task_id (FK)  │  │ • task_id (FK)      │
│ • task_statistics_id (FK)   │  │ • start         │  │ • start             │
│ • associated_task_stats_id   │  │ • end           │  │ • end               │
│   (FK, nullable)            │  │ • value         │  │ • value             │
│ • created_at, updated_at    │  │ • fixed         │  │ • fixed            │
│ • name, start, deadline    │  │ • location      │  │ • location          │
│ • difficulty, duration      │  └─────────────────┘  └──────────┬──────────┘
│ • location_id (FK)          │                                   │
│ • importance, urgency, value│                    ┌─────────────────┐
│ • fixed_time, fixed_start   │                    │schedule_changes │
│ • rated (BOOLEAN)          │                    │ • task_id (FK)   │◄──────────┘
│                             │                    │ • change_type   │
└──────────────┬──────────────┘                    │ • new_slot_*    │
                │                                   │ • created_at    │
                 ▼                                   └─────────────────┘
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WORKFLOW & DRAFT TABLES                            │
├─────────────────────────────┐  ┌───────────────────────────────────────────┐
│     unscheduled_tasks       │  │             task_drafts                   │
│                             │  │                                           │
│ • task_id (PK, FK→tasks.id) │  │ • id (PK, UUID)                           │
│ • created_at (FIFO order)   │  │ • content (JSONB) - Task + match_result   │
│                             │  │ • created_at (TIMESTAMP)                 │
└─────────────────────────────┘  └───────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NORMALIZATION TABLES                               │
├─────────────────────────────┐  ┌───────────────────────────────────────────┐
│       categories            │  │             locations                     │
│                             │  │                                           │
│ • id (UUID, PK)             │  │ • id (UUID, PK)                           │
│ • name (TEXT, UNIQUE)       │  │ • name (TEXT, UNIQUE)                     │
│ • created_at, updated_at    │  │ • created_at, updated_at                │
└──────────────┬──────────────┘  └──────────┬────────────────────────────────┘
                │                            │
┌──────────────▼──────────────┐  ┌────────▼────────┐  ┌──────────────────┐
│    task_categories           │  │task_stats_locations│  │cat_stats_locations│
│                             │  │                    │  │                  │
│ • task_id (FK, PK)          │  │ • statistics_id     │  │ • statistics_id  │
│ • category_id (FK, PK)      │  │ • location_id      │  │ • location_id    │
│ • priority (INTEGER)        │  │ • count (INTEGER) │  │ • count (INTEGER)│
└─────────────────────────────┘  └────────────────────┘  └──────────────────┘
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STATISTICS TABLES                                  │
├─────────────────────────────┐  ┌───────────────────────────────────────────┐
│     tasks_statistics        │  │         category_statistics               │
│                             │  │                                           │
│ • id (UUID, PK)             │  │ • id (UUID, PK)                           │
│ • task_name (TEXT, UNIQUE)  │  │ • category_id (UUID, UNIQUE)               │
│ • task_name_vector (FLOAT[])│  │ • avg_duration (JSONB)                  │
│ • avg_duration (JSONB)       │  │ • avg_duration_delta (JSONB)             │
│ • avg_duration_delta (JSONB) │  │ • avg_difficulty (FLOAT)                │
│ • avg_difficulty (FLOAT)    │  │ • avg_difficulty_delta (FLOAT)          │
│ • avg_difficulty_delta(FLT) │  │ • completed_count, uncompleted_count    │
│ • completed_count (INTEGER) │  │ • records (INTEGER)                       │
│ • uncompleted_count (INT)   │  │ • category_time_scores (JSONB)            │
│ • records (INTEGER)        │  │ • created_at, updated_at                │
│ • task_time_scores (JSONB)  └───────────────────────────────────────────┘
│ • created_at, updated_at                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Tables

### 4.1 `tasks` — Primary Task Storage

Source of truth for all task definitions.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | UUID | Primary key | Auto-generated via `gen_random_uuid()` |
| task_statistics_id | UUID | FK → `tasks_statistics.id` | Points to this task's own stats |
| associated_task_statistics_id | UUID | FK → `tasks_statistics.id` | Nullable. Points to matched task's stats |
| location_id | UUID | FK → `locations.id` | Normalized location reference |
| created_at | TIMESTAMP | Creation timestamp | Auto-managed by SQLAlchemy |
| updated_at | TIMESTAMP | Last update timestamp | Auto-managed by SQLAlchemy |
| name | TEXT | Task name | Cannot be empty |
| start | DATETIME | Temporal start constraint | Nullable for fixed-time |
| deadline | DATETIME | Temporal deadline | Nullable for fixed-time |
| difficulty | FLOAT | Task difficulty | Range: 0.0–1.0 |
| duration | INTEGER | Task duration in minutes | Range: 1–1439 |
| importance | FLOAT | Task importance | Range: 0.0–1.0 |
| urgency | FLOAT | Computed urgency | Range: 0.0–1.0 |
| value | FLOAT | Composite task value | Range: 0.0–1.0 |
| fixed_time | BOOLEAN | Bypass scheduler scoring | Default: false |
| fixed_start | DATETIME | Exact start time if fixed_time=true | Required if fixed_time=true |
| rated | BOOLEAN | User has rated this task | Default: false |

**Cascade Rule:** `ON DELETE CASCADE` to `main_schedule`, `provisional_schedule`, `schedule_changes`, `unscheduled_tasks`. `ON DELETE NO ACTION` to `tasks_statistics`.

**Validation Logic:**
- If `fixed_time = false`: `start` and `deadline` must be NOT NULL. `fixed_start` must be NULL.
- If `fixed_time = true`: `start` and `deadline` must be NULL. `fixed_start` must be NOT NULL.

### 4.2 `main_schedule` — Main Committed Schedule

Committed, real calendar. Source of truth for what the user sees.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| task_id | UUID | FK → `tasks.id` ON DELETE CASCADE |
| start | DATETIME | Slot start time |
| end | DATETIME | Slot end time |
| value | FLOAT | Task value at scheduling time |
| fixed | BOOLEAN | If true, cannot be displaced by scheduler |
| location | TEXT | For location continuity boost |

### 4.3 `provisional_schedule` — Working Copy

Same schema as `main_schedule`. Used by Scheduler to stage changes before commit.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| task_id | UUID | FK → `tasks.id` ON DELETE CASCADE |
| start | DATETIME | Slot start time |
| end | DATETIME | Slot end time |
| value | FLOAT | Task value |
| fixed | BOOLEAN | If true, cannot be displaced |
| location | TEXT | For location continuity boost |

### 4.4 `schedule_changes` — Change Log

Records only `insert` and `move` operations applied to transform Main → Provisional.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| provisional_schedule_slot_id | UUID | FK → `provisional_schedule.id` ON DELETE CASCADE |
| change_type | TEXT | `'insert'` or `'move'` |
| old_slot_start | DATETIME | Nullable. Previous slot start for moves |
| old_slot_end | DATETIME | Nullable. Previous slot end for moves |
| new_slot_start | DATETIME | For insert/move operations |
| new_slot_end | DATETIME | For insert/move operations |
| created_at | TIMESTAMP | When change was recorded |

---

## 5. Workflow & Draft Tables

### 5.1 `unscheduled_tasks` — Task Queue

Stores only IDs of tasks created/modified but not yet placed into any schedule.

| Field | Type | Description |
|-------|------|-------------|
| task_id | UUID | Primary key, FK → `tasks.id` ON DELETE CASCADE |
| created_at | TIMESTAMP | Used for FIFO ordering in batch scheduling |

### 5.2 `task_drafts` — Temporary Draft Storage

Stores tasks created via NLP that are not yet committed by the user.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key. Auto-generated. Returned to frontend as `draft_id` |
| content | JSONB | Full task payload + match_result |
| created_at | TIMESTAMP | Used by Garbage Collector. Drafts older than 24h are auto-deleted |

**Purpose:**
- Prevents "zombie tasks" if user abandons creation flow
- Allows frontend to edit task data without losing NLP context (vectors, associations)
- Clean separation between "pending" and "committed" states

**Content Structure:**
```json
{
    "task": {
        "name": "Math homework",
        "start": "2026-04-20T09:00:00",
        "deadline": "2026-04-25T17:00:00",
        "difficulty": 0.7,
        "duration": 60,
        "category": ["study"],
        "location": "Library",
        "importance": 0.6
    },
    "match_result": {
        "associated_id": "uuid-of-stats",
        "association_status": "same",
        "name_vector": [0.1, 0.2, ...]
    }
}
```

---

## 6. Normalization Tables

### 6.1 `categories` — Master Category List

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| name | TEXT | Unique category name (e.g., "study", "fitness") |
| created_at | TIMESTAMP | Audit timestamp |
| updated_at | TIMESTAMP | Audit timestamp |

### 6.2 `task_categories` — Task-Category Junction (Many-to-Many)

| Field | Type | Description |
|-------|------|-------------|
| task_id | UUID | FK → `tasks.id`. Part of composite PK. |
| category_id | UUID | FK → `categories.id`. Part of composite PK. |
| priority | INTEGER | Ordering priority. 1 = highest priority. |

### 6.3 `locations` — Master Location List

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| name | TEXT | Unique location name (e.g., "home", "library") |
| created_at | TIMESTAMP | Audit timestamp |
| updated_at | TIMESTAMP | Audit timestamp |

---

## 7. Statistics Tables

### 7.1 `tasks_statistics` — Task-Level Behavioral Data

Updated by Stats Recorder. Read by Enrichment, Task Matching, Scheduler.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| task_name | TEXT | Current task name. Unique constraint. |
| task_name_vector | FLOAT[] | 384-dim semantic embedding (used for matching). |
| avg_duration | JSONB | Keyed by difficulty bucket: `{"0.0": {"count": 5, "avg": 30}, "0.5": {"count": 3, "avg": 45}}` |
| avg_duration_delta | JSONB | Keyed by difficulty bucket: `{"0.5": {"count": 3, "avg": 10}}` |
| avg_difficulty | FLOAT | Running average of committed difficulty. |
| avg_difficulty_delta | FLOAT | Running average of (actual - committed). |
| completed_count | INTEGER | Successful completions. |
| uncompleted_count | INTEGER | Failed/cancelled completions. |
| records | INTEGER | Total commits (creation + modifications). |
| task_time_scores | JSONB | Radial time preferences: `{"10:00": 2.5, "10:15": 1.75}`. |
| created_at | TIMESTAMP | Audit timestamp |
| updated_at | TIMESTAMP | Audit timestamp |

**Bucket Structure (v3.0):**
```python
avg_duration = {
    "0.0": {"count": 5, "avg": 30},
    "0.5": {"count": 3, "avg": 45},
    "1.0": {"count": 4, "avg": 45}
}
```

### 7.2 `category_statistics` — Category-Level Behavioral Data

Pre-seeded with: `study`, `fitness`, `work`, `personal`.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key. Inherited from BaseModel. |
| category_id | UUID | FK → `categories.id`. Unique constraint. |
| avg_duration | JSONB | Keyed by difficulty bucket with counts: `{"0.0": {"count": 5, "avg": 30}, "0.5": {"count": 3, "avg": 45}}` |
| avg_duration_delta | JSONB | Same structure as avg_duration |
| avg_difficulty | FLOAT | Single value per category. |
| avg_difficulty_delta | FLOAT | Single value per category. |
| completed_count | INTEGER | Category-level completions. |
| uncompleted_count | INTEGER | Category-level failures. |
| records | INTEGER | Category-level commits. |
| category_time_scores | JSONB | Time preferences: `{"10:00": 1.8, "14:00": 2.2}`. |
| created_at | TIMESTAMP | Audit timestamp |
| updated_at | TIMESTAMP | Audit timestamp |

### 7.3 `task_statistics_locations` — Task-Location Junction

Tracks location usage per task for location preferences.

| Field | Type | Description |
|-------|------|-------------|
| statistics_id | UUID | FK → `tasks_statistics.id`. Part of PK. |
| location_id | UUID | FK → `locations.id`. Part of PK. |
| count | INTEGER | Number of times this location was used |

### 7.4 `category_statistics_locations` — Category-Location Junction

Tracks location usage per category.

| Field | Type | Description |
|-------|------|-------------|
| statistics_id | UUID | FK → `category_statistics.id`. Part of PK. |
| location_id | UUID | FK → `locations.id`. Part of PK. |
| count | INTEGER | Number of times this location was used |

---

## 8. Relationships & Cascade Rules

| From Table | To Table | Relationship | Foreign Key | Cascade on Delete |
|------------|----------|--------------|-------------|-------------------|
| tasks | tasks_statistics | Many-to-one | task_statistics_id | NO ACTION |
| tasks | tasks_statistics | Many-to-one | associated_task_statistics_id | NO ACTION |
| tasks | main_schedule | One-to-many | task_id | CASCADE |
| tasks | provisional_schedule | One-to-many | task_id | CASCADE |
| tasks | schedule_changes | One-to-many | task_id | CASCADE |
| tasks | unscheduled_tasks | One-to-one | id = task_id | CASCADE |
| task_categories | tasks | Many-to-one | task_id | CASCADE |
| task_categories | categories | Many-to-one | category_id | CASCADE |

---

## 9. Recommended Indexes

```sql
-- Overlap checks for Scheduler
CREATE INDEX idx_provisional_range ON provisional_schedule (start, end);
CREATE INDEX idx_main_schedule_range ON main_schedule (start, end);

-- FIFO ordering for batch scheduling
CREATE INDEX idx_unscheduled_fifo ON unscheduled_tasks (created_at);

-- Draft cleanup optimization
CREATE INDEX idx_drafts_created_at ON task_drafts (created_at);

-- Semantic matching acceleration
CREATE INDEX idx_stats_name ON tasks_statistics (task_name);

-- UI schedule fetch optimization
CREATE INDEX idx_tasks_rated ON tasks (rated);

-- Normalization lookups
CREATE INDEX idx_categories_name ON categories (name);
CREATE INDEX idx_locations_name ON locations (name);
```

---

## 10. Two Denominators

The statistics system uses two separate denominators to maintain mathematical soundness:

| Average Type | Denominator | When Updated | Fields |
|--------------|--------------|--------------|--------|
| **Plan averages** | `records` | On task commit | `avg_duration`, `avg_difficulty` |
| **Delta averages** | `completed_count` | On task rating | `avg_duration_delta`, `avg_difficulty_delta` |

---

*Document prepared for ONIA 2026.*