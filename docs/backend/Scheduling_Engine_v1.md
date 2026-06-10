# Scheduling Engine — Technical Documentation
**Version:** 2.0 (Complete)
**Last Updated:** April 26, 2026
**Competition:** ONIA 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Position in Pipeline](#2-position-in-pipeline)
3. [Architecture](#3-architecture)
4. [Constants](#4-constants)
5. [Data Structures](#5-data-structures)
6. [Algorithm Flow](#6-algorithm-flow)
7. [Constraint Solver](#7-constraint-solver)
8. [Slot Generation](#8-slot-generation)
9. [Scoring System](#9-scoring-system)
10. [Displacement Handler](#10-displacement-handler)
11. [Fixed Task Scheduling](#11-fixed-task-scheduling)
12. [Flexible Task Scheduling](#12-flexible-task-scheduling)
13. [Batch Scheduling](#13-batch-scheduling)
14. [Safety Guards](#14-safety-guards)
15. [Edge Cases](#15-edge-cases)
16. [Bug Fixes](#16-bug-fixes)
17. [API Integration](#17-api-integration)
18. [Testing Considerations](#18-testing-considerations)
19. [Performance Analysis](#19-performance-analysis)
20. [Summary](#20-summary)

---

## 1. Overview

### 1.1 What is the Scheduling Engine?

The Scheduling Engine is the **fourth stage** in the VM.AI pipeline. It takes unscheduled tasks from the queue and determines optimal calendar slots for them using a **stability-first incremental approach**.

Unlike global optimization algorithms that reshuffle the entire schedule for mathematical perfection, this engine prioritizes **predictable local changes** - when a new task is added, only minimal adjustments are made to the existing schedule.

### 1.2 Key Design Principles

| Principle | Description | Why It Matters |
|-----------|-------------|---------------|
| **Stable Incremental** | Only schedules new tasks, doesn't reschedule everything | Users expect predictable changes |
| **1-Layer Displacement** | Tasks can only displace at most 1 other task | Prevents cascade effects |
| **Value Threshold** | Can't displace unless 25% more valuable | Protects low-value tasks |
| **12s Timeout** | Hard limit on execution time | Prevents hanging |
| **All Slots Scored** | Score all viable slots (TOP_N_CANDIDATES=400) | Low-value tasks can still get scheduled |

### 1.3 What It Does

The engine performs the following operations in sequence:

1. **Fetches** unscheduled tasks from the queue (FIFO order)
2. **Analyzes** each task's constraints (start, deadline, duration)
3. **Generates** viable time windows
4. **Creates** individual time slots from windows
5. **Scores** every slot using multiple factors
6. **Attempts** placement with displacement handling
7. **Records** all changes in schedule_changes table
8. **Commits** each task individually

### 1.4 What It Does NOT Do

| What It Doesn't Do | Reason |
|-------------------|--------|
| Handle NLP parsing | Handled by Stage 1 |
| Make enrichment decisions | Handled by Stage 3 |
| Update statistics | Handled by Stage 5 (Stats Recorder) |
| Handle recurring tasks | Future scope |
| Manage user preferences directly | Via time_scores from Stats Recorder |

---

## 2. Position in Pipeline

### 2.1 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VM.AI PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: NLP Parser                                                │
│  Input: "finish chemistry homework before Friday"                   │
│  Output: TaskPayload with {value, predicted} structure            │
│                                                                      │
│  Stage 2: Task Matching                                            │
│  Input: Task name                                                  │
│  Output: {associated_id, association_status, name_vector}           │
│                                                                      │
│  Stage 3: Enrichment                                              │
│  Input: TaskPayload + Match result                                 │
│  Output: Full task data with urgency, value, importance             │
│                                                                      │
│  Stage 4: SCHEDULER (WE ARE HERE)                                │
│  Input: Enriched tasks from unscheduled_queue                      │
│  Output: provisional_schedule + schedule_changes                 │
│                                                                      │
│  Stage 5: Stats Recorder                                          │
│  Input: Completed/rated tasks                                     │
│  Output: Updated statistics tables                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Within the Scheduling Stage

```
User clicks "Schedule Tasks"
        │
        ▼
┌───────────────────┐
│  Fetch Queue     │  Get tasks from unscheduled_tasks (FIFO)
└────────┬─────────┘
         │
         ▼
┌───────────────────┐
│  For Each Task   │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│ Fixed │ │Flexibl│
│ Task  │ │ Task  │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌─────────────────────┐
│ Constraint Solver   │
│ - Time windows     │
│ - Subtract fixed   │
│ - Subtract zones   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Slot Generator      │
│ - Split windows    │
│ - Consider         │
│   duration         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Scoring             │
│ - Score all slots  │
│ - Sort by score    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Displacement        │
│ - Try each slot    │
│ - Handle conflicts │
│ - Reschedule if    │
│   needed           │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Write to DB         │
│ - provisional_      │
│   schedule         │
│ - schedule_changes │
└─────────────────────┘
```

---

## 3. Architecture

### 3.1 Class-Based Design

The Scheduling Engine is implemented as a Python class following the **Service Layer pattern**:

```python
from app.services.schedule_engine import schedule_engine

# The singleton instance
schedule_engine = ScheduleEngine()
```

### 3.2 Public Methods

| Method | Description |
|--------|-------------|
| `schedule_single(task, db)` | Schedule one task |
| `schedule_batch(db, timeout, task_ids)` | Schedule multiple tasks |

### 3.3 Private Methods

| Method | Purpose |
|--------|---------|
| `_schedule_fixed_task(task, db)` | Handle fixed-time tasks |
| `_schedule_flexible_task(task, db)` | Handle flexible tasks |
| `_get_time_windows(task, db, exclude_ranges, fixed_slots)` | Build viable time windows |
| `_subtract_fixed_tasks(windows, db)` | Remove fixed task times |
| `_subtract_dead_zones(windows)` | Remove dead zones |
| `_generate_slots(windows, duration)` | Create 15-min slots |
| `_score_slot(...)` | Calculate slot quality |
| `_handle_displacement(...)` | Resolve conflicts |
| `_try_reschedule_task(task, db, layer)` | Reschedule displaced task |
| `_place_task(...)` | Write to provisional_schedule |

### 3.4 Data Flow

```
schedule_single(task, db)
    │
    ├─→ _schedule_fixed_task() ─→ Fixed task logic
    │
    └─→ _schedule_flexible_task() ─→ Flexible task logic
            │
            ├─→ _get_time_windows()
            │       │
            │       ├─→ _subtract_fixed_tasks()
            │       └─→ _subtract_dead_zones()
            │
            ├─→ _generate_slots()
            │
            ├─→ _score_slot() ─→ Multiple boost calculations
            │       │
            │       ├─→ _get_location_boost()
            │       ├─→ _get_free_slot_boost()
            │       ├─→ _get_time_score_boost()
            │       ├─→ _get_urgency_boost()
            │       ├─→ _get_continuity_boost()
            │       └─→ _get_overlap_penalty()
            │
            └─→ _handle_displacement()
                    │
                    ├─→ _can_displace()
                    ├─→ _try_reschedule_task() ─→ Recursive for layer+1
                    └─→ _place_task()
```

---

## 4. Constants

### 4.1 Core Scheduling Constants

```python
SLOT_INTERVAL_MINUTES = 15
```

**Purpose:** Time granularity for scheduling slots.

**Details:** All slots are aligned to 15-minute boundaries. This applies to:
- Slot start times generated
- Fixed task rounding
- Duration rounding

**Example:** If a task starts at 11:48, it rounds to 11:45.

---

```python
HORIZON_DAYS = 7
```

**Purpose:** How far ahead the scheduler looks for available slots.

**Details:** The scheduling window is always 7 days from the current day. Tasks with deadlines beyond 7 days are truncated to the horizon.

**Calculation:**
- Total slots = 7 days × 56 slots/day (14 waking hours × 4) = 392 potential slots
- With 24-hour consideration: 7 × 96 = 672 possible slots

---

```python
TOP_N_CANDIDATES = 400
```

**Purpose:** Number of top-scoring slots to consider.

**Details:** Previously was 15, now uses 400 (all possible slots in 7-day window). This ensures low-value tasks can still find slots even if higher-value tasks occupy the "top" spots.

**Reason for change:** With FREE_SLOT_BOOST increased to 0.5, free slots naturally bubble to the top. Using all slots ensures no viable slot is missed.

---

```python
VALUE_THRESHOLD = 1.25
```

**Purpose:** Minimum value ratio to displace an existing task.

**Formula:**
```
can_displace = new_task.value > existing_task.value × 1.25
```

**Examples:**
| New Value | Existing Value | Can Displace? |
|-----------|--------------|---------------|
| 0.80 | 0.60 | YES (0.80 > 0.60 × 1.25 = 0.75) |
| 0.70 | 0.60 | NO (0.70 < 0.75) |
| 0.76 | 0.60 | YES (0.76 > 0.75) |
| 1.00 | 0.50 | YES (1.00 > 0.625) |

---

```python
MAX_DISPLACEMENT_LAYERS = 1
```

**Purpose:** Maximum depth of displacement chain.

**Details:** 
- Layer 0: Original task being scheduled
- Layer 1: Task displaced by original
- Layer 2: Task displaced by Layer 1 (NOT ALLOWED)

**Why 1:** Prevents cascade effects. If displacement chain is too long, users lose track of what changed.

---

```python
TIMEOUT_SECONDS = 12
```

**Purpose:** Hard timeout to prevent infinite scheduling.

**Details:** If batch scheduling exceeds 12 seconds, it stops and returns partial results. This ensures the system remains responsive.

---

### 4.2 Scoring Constants

```python
BASE_SLOT_SCORE = 1.0
```

**Purpose:** Baseline score for any slot.

**Details:** Every slot starts with 1.0. All boosts and penalties are added/subtracted from this base.

---

```python
FREE_SLOT_BOOST = 0.5
```

**Purpose:** Boost for scheduling in free (unoccupied) slots.

**Details:** Increased from 0.1 to 0.5 to make free slots more attractive. With this boost, a free slot always scores higher than an occupied slot (even after penalties).

**Example:**
- Free slot: 1.0 + 0.5 = 1.5
- Occupied slot: 1.0 - 0.15 (overlap penalty) = 0.85

---

```python
LOCATION_BASE_BOOST = 0.25
```

**Purpose:** Maximum boost for location continuity.

**Details:** Tasks scheduled near other tasks at the same location get a boost.

**Breakdown:**
- Same location before (within 2 hours): +0.125 (0.5 × 0.25)
- Same location after (within 2 hours): +0.125 (0.5 × 0.25)
- Both: +0.25 (full boost)
- Neither: 0

---

```python
TIME_SCORE_AMPLIFIER = 0.3
```

**Purpose:** Scaling factor for time preference scores.

**Details:** Time scores range from -10 to +10 from user history. This amplifier scales them to a reasonable boost range.

**Formula:**
```
boost = TIME_SCORE_AMPLIFIER × (score / 10)
```

**Examples:**
| Time Score | Boost |
|-----------|-------|
| +10 | +0.3 |
| +5 | +0.15 |
| 0 | 0 |
| -5 | -0.15 |
| -10 | -0.3 |

---

```python
URGENCY_AMPLIFIER = 0.3
```

**Purpose:** Scaling factor for urgency-based boost.

**Details:** Earlier slots in the horizon get higher boost to encourage scheduling tasks sooner rather than later.

**Formula:**
```
boost = URGENCY_AMPLIFIER × urgency_value × (1 - position_ratio)
```

Where:
- `urgency_value = (task.urgency × 0.5) + (task.importance × 0.5)`
- `position_ratio = slot_minutes_from_now / total_minutes_in_horizon`

**Examples:**
| Position in Horizon | Modifier | Boost (urgency=0.7, importance=0.6) |
|---------------------|----------|-------------------------------------|
| 0% (now) | 1.0 | 0.195 |
| 25% | 0.75 | 0.146 |
| 50% | 0.50 | 0.098 |
| 75% | 0.25 | 0.049 |
| 100% | 0.0 | 0.0 |

---

```python
CONTINUITY_BASE_BOOST = 0.1
```

**Purpose:** Boost for scheduling immediately after another task.

**Details:** Rewards scheduling tasks back-to-back.

**Breakdown:**
- 1 slot after (15 min gap): +0.05 (0.5 × 0.1)
- 2 slots after (30 min gap): +0.1 (1.0 × 0.1)
- 3 slots after (45 min gap): +0.05 (0.5 × 0.1)
- 4+ slots after: 0

---

```python
OVERLAP_BASE_PENALTY = 0.15
```

**Purpose:** Penalty per overlapping task.

**Details:** If a slot overlaps with existing tasks, each overlap reduces the score.

**Formula:**
```
penalty = OVERLAP_BASE_PENALTY × number_of_overlapping_tasks
```

**Examples:**
| Overlapping Tasks | Penalty |
|------------------|--------|
| 0 | 0 |
| 1 | -0.15 |
| 2 | -0.30 |
| 3 | -0.45 |

---

### 4.3 Dead Zones

```python
DEAD_ZONES = [
    ("23:00", "06:00"),  # Sleep time
]
```

**Purpose:** Time ranges that should never have scheduled tasks.

**Details:** 
- These are hard constraints
- Tasks cannot be scheduled during these times
- Applied during constraint solving phase

**Example:**
```
Window: 09:00 - 18:00
Dead zone: 13:00 - 15:00
Result: 09:00 - 13:00 AND 15:00 - 18:00
```

---

## 5. Data Structures

### 5.1 TimeWindow

```python
@dataclass
class TimeWindow:
    date: str           # "2026-04-20"
    start_time: str     # "09:00"
    end_time: str       # "17:00"
```

**Purpose:** Represents a continuous time range where a task can be scheduled.

**Example:**
```python
window = TimeWindow(
    date="2026-04-20",
    start_time="09:00",
    end_time="17:00"
)
```

---

### 5.2 CandidateSlot

```python
@dataclass
class CandidateSlot:
    start: datetime
    end: datetime
    score: float
```

**Purpose:** A scored time slot from the scoring phase.

**Example:**
```python
slot = CandidateSlot(
    start=datetime(2026, 4, 20, 9, 0),
    end=datetime(2026, 4, 20, 9, 45),
    score=1.35
)
```

---

### 5.3 SchedulingResult

```python
@dataclass
class SchedulingResult:
    success: bool
    task_id: Optional[UUID]
    slot_id: Optional[UUID]
    slot_start: Optional[datetime]
    slot_end: Optional[datetime]
    displaced_tasks: List[UUID]
    message: str
```

**Purpose:** Return value for single task scheduling.

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| success | bool | True if scheduled, False if failed |
| task_id | UUID | ID of scheduled task |
| slot_start | datetime | Start time (if success) |
| slot_end | datetime | End time (if success) |
| displaced_tasks | List[UUID] | Tasks moved to accommodate this one |
| message | str | Human-readable status |

---

### 5.4 BatchSchedulingResult

```python
@dataclass
class BatchSchedulingResult:
    scheduled_count: int
    failed_count: int
    unscheduled_remaining: List[UUID]
    results: List[SchedulingResult]
    execution_time_ms: int
```

**Purpose:** Return value for batch scheduling.

---

## 6. Algorithm Flow

### 6.1 Single Task Scheduling Flow

```
START: schedule_single(task, db)
       │
       ▼
┌──────────────────┐
│ Is fixed_time?    │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
   YES        NO
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────────────┐
│ Fixed   │ │ Flexible Task         │
│ Task    │ │ Scheduling            │
└────┬────┘ └──────────┬───────────┘
     │                 │
     ▼                 ▼
┌─────────────┐ ┌────────────────────┐
│ Schedule   │ │ Get Time Windows   │
│ Fixed Task │ │ (constraint solver)│
└─────┬─────┘ └────────┬───────────┘
      │                │
      ▼                ▼
┌─────────────┐ ┌─────────────────┐
│ Check      │ │ Generate Slots   │
│ Overlaps  │ │ (15-min splits) │
└─────┬─────┘ └────────┬──────────┘
      │                │
      ▼                ▼
┌─────────────┐ ┌─────────────────┐
│ Can         │ │ Score All Slots │
│ Displace?  │ │ (6 factors)     │
└─────┬─────┘ └────────┬──────────┘
      │                │
      ▼                ▼
┌─────────────┐ ┌─────────────────┐
│ Try         │ │ Sort by Score  │
│ Reschedule  │ │ (descending)   │
└─────┬─────┘ └────────┬──────────┘
      │                │
      ▼                ▼
┌─────────────┐ ┌─────────────────┐
│ Place Task  │ │ Handle           │
│ in Schedule │ │ Displacement    │
└─────┬─────┘ └────────┬──────────┘
      │                │
      └───────┬────────┘
              │
              ▼
       ┌─────────────┐
       │ Return      │
       │ Result     │
       └─────────────┘
```

---

## 7. Constraint Solver

### 7.1 Purpose

The Constraint Solver takes a task and determines all possible time windows where the task could be scheduled. It eliminates impossible slots BEFORE scoring.

### 7.2 Steps

#### Step 1: Determine Date Range

```python
task_start = task.start or now()
task_deadline = task.deadline or (task_start + 7 days)
horizon_end = now() + 7 days

# If deadline beyond horizon, truncate
if task_deadline > horizon_end:
    task_deadline = horizon_end
```

**Example:**
```
Task deadline: April 30
Horizon end: April 27
→ Deadline becomes April 27
```

#### Step 2: Create Full-Day Windows

For each day in range, create a full-day window:

```
Task: April 20 - April 22

Creates:
- April 20: 00:00 - 23:59
- April 21: 00:00 - 23:59
- April 22: 00:00 - 23:59
```

#### Step 3: Subtract Fixed Tasks

For each fixed task already scheduled, remove its time from windows:

```
Before: Window 09:00 - 18:00
Fixed task: 14:00 - 15:00

After:  09:00 - 14:00 AND 15:00 - 18:00
```

**Algorithm:**
```python
for each fixed_task in provisional_schedule:
    if fixed_task overlaps window:
        split window around fixed_task
        keep part before fixed_task
        keep part after fixed_task
```

#### Step 4: Subtract Dead Zones

Same logic as fixed tasks, but applied to all configured dead zones:

```
Window: 09:00 - 18:00
Dead zone: 13:00 - 15:00

Result: 09:00 - 13:00 AND 15:00 - 18:00
```

#### Step 5: Validate Windows

Remove any windows where end <= start (empty windows).

---

### 7.3 Example

**Input:**
- Task start: April 20, 10:00
- Task deadline: April 21, 20:00
- Fixed task: April 20, 14:00-15:00
- Dead zone: 23:00-06:00 (overnight)

**Process:**

1. **Date range:** April 20 - April 21

2. **Full-day windows:**
   - April 20: 00:00-23:59
   - April 21: 00:00-23:59

3. **Subtract fixed (14:00-15:00):**
   - April 20: 00:00-14:00, 15:00-23:59

4. **Subtract dead zone (23:00-06:00):**
   - April 20: 06:00-14:00, 15:00-23:00
   - April 21: 06:00-23:00

**Output:**
```python
{
    "2026-04-20": [
        TimeWindow("2026-04-20", "06:00", "14:00"),
        TimeWindow("2026-04-20", "15:00", "23:00")
    ],
    "2026-04-21": [
        TimeWindow("2026-04-21", "06:00", "23:00")
    ]
}
```

---

## 8. Slot Generation

### 8.1 Purpose

The Slot Generator takes time windows and creates individual 15-minute slot start times. **Critically, it accounts for task duration** - a task starting at 11:45 with 90-minute duration cannot fit in a window ending at 12:00.

### 8.2 Algorithm

```python
for each window:
    # Calculate valid end (window_end - duration)
    window_end_minutes = 13:00 = 780 minutes
    duration_minutes = 90
    valid_end = 780 - 90 = 690 minutes = 11:30
    
    # Generate slots from start to valid_end
    current = window_start
    while current <= valid_end:
        add slot at current
        current += 15 minutes
```

### 8.3 Duration Rounding

**Important:** Duration is rounded UP to fill complete 15-minute intervals:

| Original Duration | Rounded Duration | Slots Used |
|------------------|-----------------|------------|
| 30 min | 30 min | 2 |
| 37 min | 45 min | 3 |
| 40 min | 45 min | 3 |
| 45 min | 45 min | 3 |
| 60 min | 60 min | 4 |
| 90 min | 90 min | 6 |

### 8.4 Example

**Input:**
- Window: 09:00 - 13:00
- Duration: 90 minutes

**Calculation:**
```
Window end: 13:00 = 780 minutes
Duration: 90 minutes (no rounding needed)
Valid end: 780 - 90 = 690 = 11:30
```

**Slots generated:**
```
09:00, 09:15, 09:30, 09:45,
10:00, 10:15, 10:30, 10:45,
11:00, 11:15, 11:30
```

**NOT generated:** 11:45, 12:00, etc. (would exceed window)

---

## 9. Scoring System

### 9.1 Overview

Every viable slot receives a score based on multiple factors. The score determines which slot to try first.

**Formula:**
```
score = BASE_SLOT_SCORE
       + location_boost
       + free_slot_boost
       + time_score_boost
       + urgency_boost
       + continuity_boost
       - overlap_penalty
```

### 9.2 Location Boost

**Purpose:** Encourage grouping tasks at the same location.

**Algorithm:**
1. Query provisional_schedule for tasks within 2 hours before slot
2. Query provisional_schedule for tasks within 2 hours after slot
3. If task at same location before: +0.125 (0.5 × 0.25)
4. If task at same location after: +0.125 (0.5 × 0.25)
5. Maximum: 0.25

**Important Bug Fix:** The "after" query must use `slot_end`, not `slot_start`:
```python
# CORRECT (fixed):
after = query(ProvisionalSlot.start >= slot_end, ...)

# WRONG (previous):
after = query(ProvisionalSlot.start >= slot_start, ...)
```

**Why?** If we're scheduling at 10:00-11:30, and another task starts at 10:30, they overlap. We shouldn't get continuity boost for an overlapping task.

---

### 9.3 Free Slot Boost

**Purpose:** Prefer unoccupied slots.

**Algorithm:**
```python
overlapping = get_overlapping_tasks(slot_start, slot_end)
if not overlapping:
    return FREE_SLOT_BOOST  # 0.5
return 0
```

---

### 9.4 Time Score Boost

**Purpose:** Incorporate user's historical time preferences.

**Lookup Priority:**
1. Task-level: `task_statistics.task_time_scores[time_slot]`
2. Associated task: `task_statistics.associated_task_statistics_id.time_scores[time_slot]`
3. Category-level: Loop through task's categories
4. Default: 0

**Algorithm:**
```python
time_key = slot.strftime("%H:%M")  # "09:00"

# Try task statistics
if task.task_statistics_id:
    stats = query(TaskStatistics).get(task.task_statistics_id)
    if stats and time_key in stats.task_time_scores:
        score = stats.task_time_scores[time_key]
        return TIME_SCORE_AMPLIFIER * (score / 10)

# Try associated statistics
...

# Try categories
...

return 0
```

---

### 9.5 Urgency Boost

**Purpose:** Encourage scheduling tasks earlier rather than later.

**Algorithm:**
```python
total_minutes = HORIZON_DAYS * 24 * 60  # 10080
now = datetime.utcnow()

# Slot cannot be in the past
if slot_start <= now:
    return -1.0

# Calculate position in horizon
slot_minutes = (slot_start - now).total_seconds() / 60
position_ratio = slot_minutes / total_minutes  # 0.0 to 1.0

# Urgency value
urgency_value = (task.urgency or 0.5) * 0.5 + (task.importance or 0.5) * 0.5

# Linear decay
boost = URGENCY_AMPLIFIER * urgency_value * (1 - position_ratio)
```

**Important Bug Fix:** Must use absolute time difference, not midnight-based:
```python
# CORRECT (fixed):
slot_minutes = (slot_start - now).total_seconds() / 60

# WRONG (previous):
slot_minutes = (slot_start - now.replace(hour=0, minute=0)).total_seconds() / 60
```

**Why?** The previous code only counted minutes within the current day, not total time from now. For slots on day 2+, it gave wrong position ratios.

---

### 9.6 Continuity Boost

**Purpose:** Reward scheduling immediately after another task.

**Algorithm:**
```python
prev_task = query(ProvisionalSlot)
    .filter(ProvisionalSlot.end <= slot_start)
    .order_by(ProvisionalSlot.end.desc())
    .first()

if not prev_task:
    return 0

# Calculate gap in slots
gap_slots = round((slot_start - prev_task.end).total_seconds() / 60 / 15)

if gap_slots == 1:  # 15 min gap
    return CONTINUITY_BASE_BOOST * 0.5  # 0.05
elif gap_slots == 2:  # 30 min gap
    return CONTINUITY_BASE_BOOST * 1.0  # 0.1
elif gap_slots == 3:  # 45 min gap
    return CONTINUITY_BASE_BOOST * 0.5  # 0.05
else:
    return 0
```

---

### 9.7 Overlap Penalty

**Purpose:** Discourage scheduling in crowded time periods.

**Algorithm:**
```python
overlapping = get_overlapping_tasks(slot_start, slot_end)
return OVERLAP_BASE_PENALTY * len(overlapping)
```

---

### 9.8 Complete Scoring Example

**Task:** urgency=0.8, importance=0.6, location="School"  
**Slot:** Day 1, 10:00 (early in horizon)  
**Condition:** Free slot, no overlaps, nearby task at "School"

**Calculation:**
```
base = 1.0
location_boost = 0.25 (same location on both sides)
free_slot_boost = 0.5
time_score_boost = 0.0 (no history)
urgency_boost = 0.3 × 0.7 × (1 - 0.01) ≈ 0.208
continuity_boost = 0.1 (30 min gap)
overlap_penalty = 0

TOTAL = 1.0 + 0.25 + 0.5 + 0.0 + 0.208 + 0.1 - 0 = 2.058
```

---

## 10. Displacement Handler

### 10.1 Purpose

When a desired slot is occupied, the displacement handler decides whether to:
- Skip the slot
- Displace the existing task(s)
- Abort scheduling entirely

### 10.2 Algorithm

```python
def _handle_displacement(task, slot_start, slot_end, db, layer=1):
    # 1. Find overlapping tasks
    overlapping = get_overlapping_tasks(slot_start, slot_end)
    
    # 2. No overlap = easy, place directly
    if not overlapping:
        return place_task(task, slot_start, slot_end)
    
    # 3. Check for fixed tasks (cannot displace)
    if any(t.fixed for t in overlapping):
        return failure("Slot occupied by fixed task")
    
    # 4. Sort by value (lowest first)
    sorted_tasks = sorted(overlapping, key=lambda t: t.value)
    
    # 5. Try to displace each
    for existing in sorted_tasks:
        # Check value threshold
        if not can_displace(task, existing):
            return failure("Value threshold not met")
        
        # Try to reschedule displaced task
        reschedule_result = try_reschedule(existing.task, layer + 1)
        
        if not reschedule_result.success:
            return failure("Cannot reschedule displaced task")
        
        # Record the move
        record_move(existing, reschedule_result)
    
    # 6. All displaced successfully, place new task
    return place_task(task, slot_start, slot_end)
```

### 10.3 Value Threshold Check

```python
def can_displace(new_task, existing):
    return new_task.value > existing.value * VALUE_THRESHOLD
    # i.e., new > existing × 1.25
```

### 10.4 Layer Tracking

Each displacement increments the layer counter:

- **Layer 0:** Original task being scheduled
- **Layer 1:** Task displaced by Layer 0
- **Layer 2+:** NOT ALLOWED (max is 1)

If a displaced task needs to displace another, it fails if layer would exceed MAX_DISPLACEMENT_LAYERS.

### 10.5 Example Flow

**Scenario:**
- New task (value 0.9) wants slot 10:00-11:00
- Existing task (value 0.6) occupies 10:00-11:00

**Process:**
1. Find overlapping: task at 10:00-11:00 (value 0.6)
2. Is it fixed? No → continue
3. Value check: 0.9 > 0.6 × 1.25 = 0.75 → YES
4. Delete old slot, try to reschedule task (value 0.6)
5. If reschedule succeeds → place new task
6. If reschedule fails → try next slot in list

---

## 11. Fixed Task Scheduling

### 11.1 What is a Fixed Task?

A fixed task has a specific time that must be honored:
- `fixed_time = True`
- `fixed_start` set to desired time
- Duration specifies length

### 11.2 Algorithm

```python
def schedule_fixed_task(task, db):
    # 1. Round start down to 15-min interval
    slot_start = round_down(task.fixed_start, 15)
    
    # 2. Calculate end from ORIGINAL start (not rounded), then round up
    raw_end = task.fixed_start + task.duration
    slot_end = round_up(raw_end)
    
    # 3. Check for conflicts
    overlapping = get_overlapping_tasks(slot_start, slot_end)
    
    for each overlapping task:
        if overlapping.fixed:
            return failure("Cannot displace fixed task")
        
        # Try to reschedule flexible task (no value threshold for fixed→flexible)
        if not reschedule(overlapping):
            return failure("Cannot reschedule displaced task")
    
    # 4. Place fixed task
    return place_task(task, slot_start, slot_end)
```

### 11.3 Important: End Time Calculation

**Fixed tasks calculate end differently than flexible:**

```python
# Example: fixed_start=13:40, duration=30

# Step 1: Round start DOWN
fixed_start = 13:40 → 13:30

# Step 2: Calculate from ORIGINAL start, not rounded
raw_end = 13:40 + 30 min = 14:10

# Step 3: Round END UP
slot_end = 14:10 → 14:15

# Result: 13:30 - 14:15
```

**Why?** Users expect fixed tasks to be exactly at their specified time. The rounding ensures slots align with the 15-minute grid while preserving the user's intended duration.

---

## 12. Flexible Task Scheduling

### 12.1 What is a Flexible Task?

A flexible task doesn't have a fixed time - the scheduler finds the optimal slot.

### 12.2 Algorithm

```python
def schedule_flexible_task(task, db):
    # 1. Get viable time windows
    windows = get_time_windows(task, db)
    
    # 2. Generate slots from windows
    slots = generate_slots(windows, task.duration)
    
    # 3. Score all slots
    scored = []
    for slot in slots:
        score = score_slot(slot, task, db)
        scored.append(CandidateSlot(slot, score))
    
    # 4. Sort by score (descending)
    scored.sort(key=lambda s: s.score, reverse=True)
    
    # 5. Try top candidates (up to TOP_N_CANDIDATES)
    for candidate in scored[:TOP_N_CANDIDATES]:
        result = handle_displacement(task, candidate.start, candidate.end)
        if result.success:
            return result
    
    # 6. All failed
    return failure("No viable slot found")
```

---

## 13. Batch Scheduling

### 13.1 Purpose

Schedule multiple tasks in sequence, handling failures gracefully.

### 13.2 Algorithm

```python
def schedule_batch(db, timeout=12, task_ids=None):
    start = now()
    
    # Get tasks (from queue or provided list)
    if task_ids:
        tasks = query(Task).filter(id.in_(task_ids))
    else:
        tasks = query(UnscheduledTask).order_by(created_at)
    
    results = []
    
    for task in tasks:
        # Schedule single task
        result = schedule_single(task, db)
        results.append(result)
        
        if result.success:
            scheduled_count += 1
            if not task_ids:
                delete_from_queue(task)
        else:
            failed_count += 1
        
        # Check timeout
        if elapsed > timeout:
            break
    
    return BatchSchedulingResult(...)
```

### 13.3 Hybrid Design

The batch function supports two modes:

| Mode | Usage |
|------|-------|
| Queue mode | `schedule_batch(db)` - schedules from `unscheduled_tasks` |
| ID mode | `schedule_batch(db, task_ids=[...])` - schedules specific tasks |

This provides flexibility for testing and partial scheduling.

---

## 14. Safety Guards

### 14.1 Timeout

```python
if elapsed > TIMEOUT_SECONDS:
    break  # Return partial results
```

### 14.2 Value Threshold

```python
if not can_displace(new_task, existing):
    return failure("Value threshold not met")
```

### 14.3 Max Layers

```python
if layer > MAX_DISPLACEMENT_LAYERS:
    return failure("Max displacement layers exceeded")
```

### 14.4 Fixed Task Conflict

```python
if overlapping.fixed:
    return failure("Cannot displace fixed task")
```

### 14.5 Past Slot Check

```python
if slot_start <= now():
    logger.warning("Attempted to schedule in past")
    return -1.0  # Negative score (won't be chosen)
```

---

## 15. Edge Cases

### 15.1 Task with No Viable Windows

**Scenario:** Task deadline before current time, or deadline too soon.

**Result:** Returns failure message "No viable time windows"

---

### 15.2 All Slots Occupied by Fixed Tasks

**Scenario:** Fixed tasks fill all available time.

**Result:** Returns failure, task remains in queue.

---

### 15.3 Displacement Chain Fails

**Scenario:** Task A displaces Task B, but Task B cannot find a slot.

**Result:** 
- Task A's scheduling fails
- Task B returns to original slot
- No changes committed

---

### 15.4 Task Already in Provisional Schedule

**Scenario:** Task is being rescheduled (already has a slot).

**Result:** 
- Old slot deleted
- New slot created
- change_type = "move"

---

### 15.5 Duration Exceeds Window

**Scenario:** Task duration (90 min) > available window (30 min).

**Result:** Slot generation produces no slots, task not scheduled.

---

### 15.6 Deadline Beyond Horizon

**Scenario:** Task deadline is 15 days away.

**Result:** Deadline truncated to horizon (7 days). Task scheduled within first 7 days only.

---

## 16. Bug Fixes

### 16.1 Fixed Task End Calculation

**Previous (Wrong):**
```python
fixed_start = round_down(original_start)
fixed_end = calculate_end(fixed_start, duration)
# 13:40 → 13:30, end = 13:30 + 45 = 14:15
# But user expected end based on 13:40 + 30 = 14:10 → 14:15
```

**Fixed:**
```python
fixed_start = round_down(original_start)
fixed_end = round_up(original_start + duration)
# 13:40 → 13:30 (start)
# raw_end = 13:40 + 30 = 14:10 → 14:15 (end)
```

---

### 16.2 Location Boost After Query

**Previous (Wrong):**
```python
after = query(ProvisionalSlot.start >= slot_start, ...)
```

**Fixed:**
```python
after = query(ProvisionalSlot.start >= slot_end, ...)
```

---

### 16.3 Urgency Boost Time Calculation

**Previous (Wrong):**
```python
slot_minutes = (slot_start - today_midnight).total_seconds() / 60
```

**Fixed:**
```python
slot_minutes = (slot_start - now).total_seconds() / 60
```

---

### 16.4 Displacement Rescheduling

**Previous (Wrong):**
```python
# Just deleted existing task
db.delete(existing)
record_move(existing, new_slot=current_slot)  # Wrong!
```

**Fixed:**
```python
# Actually reschedule the displaced task
reschedule_result = try_reschedule_task(displaced_task, layer + 1)
if not reschedule_result.success:
    return failure()
record_move(existing, new_slot=reschedule_result.slot)
```

---

### 16.5 Insert vs Move Detection

**Previous (Wrong):**
```python
change_type = "insert" if not displaced_ids else "move"
```

**Fixed:**
```python
existing = query(ProvisionalSlot.task_id == task.id).first()
change_type = "move" if existing else "insert"
```

---

## 17. API Integration

### 17.1 Endpoint

```python
@router.post("/schedule/batch")
def schedule_batch(
    db: Session = Depends(get_db),
):
    return schedule_engine.schedule_batch(db)
```

### 17.2 Response Format

```json
{
    "scheduled_count": 5,
    "failed_count": 2,
    "results": [
        {
            "success": true,
            "task_id": "uuid",
            "slot_start": "2026-04-20T10:00:00",
            "slot_end": "2026-04-20T10:45:00",
            "displaced_tasks": ["uuid"],
            "message": "Task scheduled successfully"
        }
    ],
    "execution_time_ms": 3500
}
```

---

## 18. Testing Considerations

### 18.1 Unit Tests

| Component | Test Cases |
|-----------|-----------|
| Constraint Solver | Window generation, fixed task subtraction, dead zone subtraction |
| Slot Generation | Duration fitting, edge cases |
| Scoring | Each boost independently |
| Displacement | Value threshold, layer limits, reschedule failure |
| Fixed Tasks | End calculation, conflict handling |

### 18.2 Integration Tests

| Scenario | Expected |
|----------|----------|
| Single task scheduling | Task in provisional_schedule |
| Batch scheduling | All tasks scheduled or proper failure counts |
| Fixed task with conflict | Proper displacement or failure |
| Timeout | Partial results returned |

---

## 19. Performance Analysis

### 19.1 Time Complexity

```
N = number of tasks in queue
S = number of viable slots (max 400)
D = displacement depth (max 1)

Per task: O(S × D)
Per batch: O(N × S × D)

With N=20, S=400, D=1:
Worst case: 20 × 400 = 8,000 operations
```

### 19.2 Real-World Performance

For typical use:
- 5-10 tasks per batch
- 50-100 viable slots per task
- Average: <100ms per task
- Total batch: <2 seconds

**Timeout of 12s is very generous.**

---

## 20. Summary

### 20.1 Key Characteristics

| Aspect | Value |
|--------|-------|
| Architecture | Class-based (ScheduleEngine) |
| Algorithm | Constraint Solving + Scoring + Displacement |
| Complexity | O(N × 400) |
| Timeout | 12 seconds |
| Displacement | 1-layer max |
| Value Threshold | 25% |

### 20.2 Scoring Formula

```
score = 1.0 + location(0-0.25) + free(0/0.5) + time(-0.3 to +0.3) + urgency(0-0.21) + continuity(0-0.1) - overlap(0 to -∞)
```

### 20.3 Constants Summary

| Constant | Value |
|----------|-------|
| SLOT_INTERVAL_MINUTES | 15 |
| HORIZON_DAYS | 7 |
| TOP_N_CANDIDATES | 400 |
| VALUE_THRESHOLD | 1.25 |
| MAX_DISPLACEMENT_LAYERS | 1 |
| TIMEOUT_SECONDS | 12 |
| FREE_SLOT_BOOST | 0.5 |
| LOCATION_BASE_BOOST | 0.25 |
| TIME_SCORE_AMPLIFIER | 0.3 |
| URGENCY_AMPLIFIER | 0.3 |
| CONTINUITY_BASE_BOOST | 0.1 |
| OVERLAP_BASE_PENALTY | 0.15 |
| BASE_SLOT_SCORE | 1.0 |

---

*Document prepared for ONIA 2026. Updated April 26, 2026.*