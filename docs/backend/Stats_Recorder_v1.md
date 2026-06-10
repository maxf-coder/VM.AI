# Stats Recorder — Technical Documentation
**Version:** 1.0 (Final)
**Last Updated:** April 18, 2026
**Competition:** ONIA 2026

---

## 1. Overview

The Stats Recorder is the fifth stage in the VM.AI pipeline. It synchronously updates behavioral statistics when tasks are committed or rated.

### Key Design Principles
- **Synchronous Execution**: Runs in same DB transaction as triggering action to eliminate race conditions
- **Two Denominators**: Separates plan averages (records) from delta averages (completed_count)
- **Radial Decay**: Time preferences decay over time blocks to adapt to schedule changes

### What It Does
- Updates task statistics when tasks are committed (planned)
- Updates task statistics when tasks are rated (completed)
- Updates category statistics for all task categories
- Handles location tracking
- Applies radial decay to time preferences

### What It Does NOT Do
- Handle scheduling decisions
- Make enrichment or matching decisions
- Run asynchronously (for MVP)

---

## 2. Position in Pipeline

```
Task Created → Task Matching → Enrichment → Scheduler
                                    ↓
                             Stats Recorder (on commit)
                                    ↓
                             Statistics Updated
```

Also triggered separately when user rates a task:
```
User Rates Task → Stats Recorder (on rate)
                        ↓
              Statistics Updated
```

---

## 3. Two Denominators

The system uses two separate denominators to maintain mathematical soundness:

| Average Type | Denominator | When Updated |
|--------------|--------------|--------------|
| **Plan averages** | `records` | On task commit |
| **Delta averages** | `completed_count` | On task rating |

### Plan Averages (on commit)
```python
new_avg = (old_avg * records + new_value) / (records + 1)
records += 1
```

### Delta Averages (on rating)
```python
new_delta_avg = (old_delta_avg * completed_count + delta) / (completed_count + 1)
completed_count += 1
```

---

## 4. Update Triggers

### 4.1 On Task Commit
When a task is created or modified:

| Field | Update |
|-------|--------|
| `avg_duration[bucket]` | Recalculate with weighted average |
| `avg_difficulty` | Recalculate with weighted average |
| `avg_duration_delta` | NOT updated (no actual yet) |
| `avg_difficulty_delta` | NOT updated (no actual yet) |
| `records` | Increment by 1 |

### 4.2 On Task Rating
When user rates a completed task:

| Field | Update |
|-------|--------|
| `avg_duration_delta[bucket]` | Recalculate with weighted average |
| `avg_difficulty_delta` | Recalculate with weighted average |
| `completed_count` | Increment by 1 |

When user rates an incomplete task:

| Field | Update |
|-------|--------|
| `uncompleted_count` | Increment by 1 |

---

## 5. Duration Bucket Logic

Duration is bucketed by difficulty (0.0, 0.5, 1.0):

```
bucket = round(difficulty * 2) / 2
```

Example:
- Difficulty 0.7 → bucket 0.5
- Difficulty 0.3 → bucket 0.0

The Stats Recorder updates the correct bucket based on the actual or committed difficulty.

---

## 6. Location Tracking

### Task Statistics Locations
```python
# Increment count for location
tasks_statistics_locations[location_id].count += 1
```

### Category Statistics Locations
```python
# Increment count for each category's location
category_statistics_locations[category_id][location_id].count += 1
```

---

## 7. Time Score Decay

Time preferences decay over time to prevent stale scores:

```python
# In background cleanup loop (cleanup.py), runs every 24h:
TIME_SCORE_DECAY_FACTOR = 0.99
TIME_SCORE_MIN_THRESHOLD = 0.1

for task_stats in all_task_statistics:
    if task_stats.task_time_scores:
        for time_slot, score in task_stats.task_time_scores.items():
            score *= TIME_SCORE_DECAY_FACTOR
            if score < TIME_SCORE_MIN_THRESHOLD:
                score = 0.0
            task_stats.task_time_scores[time_slot] = score
```

Decay is applied globally every 24 hours — all time scores are multiplied by 0.99, and values below 0.1 are zeroed out.

---

## 8. API Endpoint

### POST /tasks/{id}/rate

**Request:**
```json
{
    "completed": true,
    "actual_duration": 75,
    "actual_difficulty": 0.8
}
```

**Validation:**
- If `completed=true`: `actual_duration` and `actual_difficulty` required
- If `completed=false`: Cannot send actual values

**Response:**
```json
{
    "success": true,
    "task_id": "uuid"
}
```

### Time Score Updates

The `update_time_score()` method automatically adjusts time preferences in multiple scenarios:

| Trigger | Boost Value | Context |
|---------|-------------|---------|
| Task rated — completed | `+0.5` | User completed the task on time |
| Task rated — uncompleted | `-0.5` | User failed to complete the task |
| Task updated from main_schedule | `-1.0` | Task removed from schedule |
| Task updated from provisional | `-2.0` | Task removed from provisional |
| Batch scheduling | `+1.0` | Task successfully scheduled |
| Provisional commit | `+2.0` | Task moved from provisional to main |

Time scores are clamped to `[-10.0, 10.0]` with step size `0.25`.

**Internal Constants:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `RECORDS_NR_TRACK` | 30 | Max records before recalculating rolling average |
| `TIME_SCORE_CLAMP` | `(-10.0, 10.0)` | Time score bounds |
| `TIME_SCORE_STEP` | `0.25` | Increment/decrement step |

---

## 9. Summary

| Aspect | Description |
|--------|-------------|
| **Purpose** | Update behavioral statistics |
| **Execution** | Synchronous (same transaction) |
| **Denominators** | records (plan), completed_count (delta) |
| **Triggers** | Task commit, Task rating |
| **Location Tracking** | Via junction tables |
| **Time Score Decay** | Applied globally (x0.99 every 24h) |

---

*Document prepared for ONIA 2026.*