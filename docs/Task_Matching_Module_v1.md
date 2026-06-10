# Task Matching Module — Technical Documentation
**Version:** 1.0 (Final)
**Last Updated:** April 18, 2026
**Competition:** ONIA 2026

---

## 1. Overview

The Task Matching module is the second stage in the VM.AI pipeline. Its job is to compare the task name parsed by the NLP module against every task name stored in the user's `tasks_statistics` table and determine whether the new task is the **same** as an existing one, **similar**, or **entirely new**.

This distinction is critical because the Enrichment module uses the match result to decide:
- Which `associated_task_statistics_id` to store in the `tasks` table
- Where to read historical averages from (matched task's row vs. category row)

### What It Does
- Encodes the parsed task name into a 384-dimensional semantic vector
- Performs exact case-insensitive string matching first
- Falls back to cosine similarity against all stored task vectors
- Classifies the relationship using fixed thresholds
- Returns a lightweight, deterministic payload for Enrichment

### What It Does NOT Do
- Modify task data or update the database
- Access user history, completion rates, or behavioral profiles
- Make scheduling, enrichment, or stats-recording decisions
- Require fine-tuning or training — used entirely off-the-shelf

---

## 2. Position in Pipeline

```
User Input
    ↓
NLP Parser → TaskPayload
    ↓
Task Matching Model → { name_vector, associated_id, association_status }
    ↓
Enrichment Module → reads tasks_statistics or category_statistics
    ↓
Enrichment → creates tasks_statistics row (if needed)
    ↓
Enrichment → creates tasks row with both ID fields
    ↓
Enrichment → inserts task_id into unscheduled_tasks
```

---

## 3. Model Details

| Property | Value |
|----------|-------|
| Model Name | `paraphrase-MiniLM-L6-v2` |
| Framework | HuggingFace `sentence-transformers` |
| Vector Size | 384 dimensions |
| Inference Speed | ~5ms per sentence on CPU |
| Disk Size | ~90MB |
| Training Required | No — used off-the-shelf |
| ONIA Compliance | Yes — fully open-source, documented |

---

## 4. Matching Algorithm

The matcher follows a strict, deterministic two-step process to maximize speed and accuracy.

### Step 1: Exact String Match (Pre-filter)
- Case-insensitive, whitespace-trimmed comparison against all `task_name` values in `tasks_statistics`
- If a match is found → immediately return `association_status: "same"`
- **Why first?** Eliminates unnecessary vector computation for exact duplicates (~30% of user inputs)

### Step 2: Semantic Similarity (Cosine Distance)
If no exact match:
1. Encode parsed task name → `input_vector` (shape: `[384]`)
2. Compute cosine similarity against every stored `task_name_vector` in `tasks_statistics`
3. Identify highest similarity score (`best_score`)
4. Classify using fixed thresholds:

| Threshold | Result |
|-----------|--------|
| `>= 0.90` | `"same"` |
| `0.60 - 0.89` | `"similar"` |
| `< 0.60` | `"none"` |

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
```

---

## 5. Output Schema

The model always returns exactly three fields:

| Field | Type | Description |
|-------|------|-------------|
| `name_vector` | `float[384]` | Encoded vector of parsed task name |
| `associated_id` | `UUID | null` | `tasks_statistics.id` of best matching task |
| `association_status` | `"same" \| "similar" \| "none"` | Classification result |

```json
{
    "name_vector": [0.23, -0.11, 0.45, 0.89, ...],
    "associated_id": "550e8400-e29b-41d4-a716-446655440000",
    "association_status": "same"
}
```

---

## 6. Match Cases & Enrichment Handoff

| `association_status` | New `tasks_statistics` row? | `task_statistics_id` | `associated_task_statistics_id` | Data Source |
|----------------------|-----------------------------|----------------------|----------------------------------|------------------------|
| `"same"` | **No** | `= associated_id` | `= associated_id` | Matched task's row (if `records >= 3`) |
| `"similar"` | **Yes** | New UUID | `= associated_id` | Matched task's row (if `records >= 3`) |
| `"none"` | **Yes** | New UUID | `null` | Category-level statistics |

> **Critical Invariant:** `associated_id` is always a `tasks_statistics.id`, never a `tasks.id`.

---

## 7. Threshold Configuration

| Constant | Default Value | Meaning |
|----------|---------------|---------|
| `EXACT_THRESHOLD` | `0.90` | `>= 0.90` → `"same"` |
| `SIMILAR_THRESHOLD` | `0.60` | `0.60–0.89` → `"similar"` |
| Fallback | `< 0.60` | `"none"` |

**Tuning Guidance:**
- Too high → fails to recognize paraphrases (`"chem hw"` vs `"chemistry homework"`)
- Too low → incorrectly merges unrelated tasks (`"gym workout"` vs `"buy groceries"`)

---

## 8. Database Reference

The Task Matching module only **reads** from `tasks_statistics`.

| Table | Field | Purpose |
|-------|-------|---------|
| `tasks_statistics` | `id` | Returned as `associated_id` |
| `tasks_statistics` | `task_name` | Used for exact string pre-filter |
| `tasks_statistics` | `task_name_vector` | 384-dim embedding for cosine similarity |

> The `tasks_statistics` table also contains statistical fields (`avg_duration`, `records`, etc.), but the matcher **ignores them completely**.

---

## 9. Cold Start Behavior

When a new user has zero task history:
- `tasks_statistics` table is empty
- Exact match loop returns no results
- Cosine similarity list is empty → `best_score = 0.0`
- Returns: `association_status: "none"`, `associated_id: null`, `name_vector: [computed]`
- Enrichment falls back to category-level statistics immediately
- System works from first input. No warmup period required.

---

## 10. Implementation Notes

### Thresholds
Hardcoded constants for classification:
```python
EXACT_THRESHOLD = 0.90
SIMILAR_THRESHOLD = 0.60
```

### Batch Similarity Optimization
Use `sentence-transformers.util.cos_sim` for vectorized comparison:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(task_names)
cosine_scores = util.cos_sim(input_embedding, embeddings)
```

### Fallback on Model Load Failure
If `sentence-transformers` fails to initialize, return `association_status: "none"` with logged warning to guarantee pipeline never blocks.

---

## 11. Summary

| Aspect | Description |
|--------|-------------|
| **Purpose** | Classify new task name against existing history |
| **Model** | `paraphrase-MiniLM-L6-v2` (384-dim, off-the-shelf) |
| **Matching Order** | Exact string → Cosine similarity → Threshold |
| **Thresholds** | `>= 0.90` = same, `0.60-0.89` = similar, `< 0.60` = none |
| **Output** | `{ name_vector, associated_id, association_status }` |
| `associated_id` Type | `tasks_statistics.id` (never `tasks.id`) |
| **DB Interaction** | Read-only on `tasks_statistics` |
| **Cold Start** | Works from first input |
| **Execution** | Synchronous, lightweight (`<10ms`) |
| **Next Stage** | Enrichment Module |

---

*Document prepared for ONIA 2026.*