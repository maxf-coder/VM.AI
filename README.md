# VM.AI — AI-Powered Personal Scheduling System

VM.AI is an AI-driven personal scheduling system that transforms natural language and image inputs into optimized, behavior-aware calendar schedules. The system uses a 7-stage pipeline: NLP Parser (T5-base), Image Classifier (EfficientNet-B4), Task Matching (MiniLM), Enrichment (statistics-based overwrites), Scheduling Engine (stable incremental), Duration Predictor (XGBoost), and Stats Recorder (behavioral learning) — together extracting and predicting attributes such as category, difficulty, importance, duration, deadline, location, and recurrence patterns.

## Problem Description

The project addresses the challenge of converting free-form natural language input into structured task schemas. Users can describe tasks in plain language (e.g., "gym every Monday at 6am", "finish report by Friday"), and the system parses this input into organized data that can be used for scheduling and task management.

Beyond NLP, the system also supports image-based task classification (EfficientNet-B4) and XGBoost-based duration prediction, enabling richer input modalities and more accurate scheduling estimates.

This aligns with ONIA competition requirements by providing a practical solution that uses AI to solve a real-world problem in task planning and time management.

## Problem Actuality

Time management and task planning are universal challenges. Studies consistently show that professionals lose a significant portion of their workweek to scheduling, coordination, and low-value organizational tasks — time that could be spent on meaningful, productive work.

| Statistic | Source |
|---|---|
| **4.4 hours/week** per person spent on scheduling activities (proposing times, checking availability, rescheduling, follow-ups) | [SkipUp / Doodle State of Meetings 2019, Reclaim.ai 2023](https://blog.skipup.ai/scheduling-salary-cost-of-manual-meeting-coordination/) |
| **61%** of the workday is spent on "work about work" — searching for information, checking messages, duplicating effort — leaving only **24%** for skilled, mission-critical tasks | [Asana Anatomy of Work Report](https://asana.com/resources/anatomy-of-work) |
| **$588 billion/year** lost by U.S. companies due to workplace interruptions and task-switching | [Hubstaff / Reuters](https://hubstaff.com/blog/time-management-statistics/) |
| **82%** of people lack an effective time management system; **51%** of the workday is spent on low-to-no-value tasks | [Hubstaff / Zippia](https://hubstaff.com/blog/time-management-statistics/) |
| **581 hours/person/year** lost to distractions — equivalent to **28%** of total working hours, costing **~$34,448/employee/year** | [The Economist Intelligence Unit](https://impact.economist.com/new-globalisation/in-search-of-lost-focus-2020/downloads/EIU_Dropbox_focus_executive_summary.pdf) |
| **47%** of college students cite time management as their top academic challenge; **80%** procrastinate regularly | [Kahoot! Study Habits Snapshot 2024](https://kahoot.com/press/2024/10/29/study-habits-snapshot-2024/) |
| **73%** of Moldovan employees consider time management a daily challenge; **68%** procrastinate, losing an average of **1.5 hours/day** | [aboutmoldova.md 2025–2026](https://aboutmoldova.md/ro/view_articles_post.php?id=2802) |
| **52%** of Moldovan employees do not plan their day; only **34%** use any digital organization tool | [aboutmoldova.md](https://aboutmoldova.md/ro/view_articles_post.php?id=2393) |

The cumulative cost is staggering. A professional spending 4.4 hours per week on manual scheduling loses roughly **229 hours per year** — nearly **6 full workweeks** — to coordination overhead alone. For an organization of 50 people, this translates to hundreds of thousands of dollars in lost productivity annually.

### What VM.AI Solves

VM.AI addresses this problem by **automating the entire task intake and scheduling pipeline**:

- **From natural language to structured schedule in one step** — users describe tasks in plain language ("gym every Monday at 6am", "finish report by Friday") and the system parses them into organized, calendar-ready entries.
- **No more manual data entry** — category, difficulty, importance, duration, deadline, location, and recurrence are all inferred automatically by 4 ML models working together.
- **Image-based task input** — snap a photo of a whiteboard, flyer, or document and the system classifies it into a task category.
- **Behavior-adaptive scheduling** — the system learns from user patterns over time, improving predictions for duration and priority.

### Measurable Impact

Based on existing studies:

- **Target:** Reduce task intake time by **50–70%** compared to manual calendar entry (e.g., typing into Google Calendar or a todo app).
- **Target:** Eliminate **4+ hours/week** of scheduling overhead per user by replacing back-and-forth coordination with a single natural language input.
- **Target:** Improve schedule adherence through accurate duration predictions and personalized difficulty/importance scoring.

## Team

- Golban Ion
- Furculita Maxim

## Project Structure

```
VM.AI/
├── src/
│   ├── parser/           # NLP parser module (training + inference)
│   ├── backend/          # FastAPI backend
│   │   ├── app/
│   │   │   ├── api/v1/endpoints/  # REST endpoints (tasks, schedule, provisional, stats, duration)
│   │   │   ├── models/            # SQLAlchemy ORM models
│   │   │   ├── schemas/           # Pydantic request/response schemas
│   │   │   ├── services/          # Business logic (parser, matching, enrichment, scheduler, stats, duration, img_to_prompt)
│   │   │   ├── utils/             # Model loader, cleanup loop, task persistence, normalization
│   │   │   ├── core/              # Config, database, logging
│   │   │   └── main.py            # FastAPI entrypoint
│   │   └── tests/                 # Backend API tests
│   └── app/              # React frontend (npm run dev)
├── models/
│   ├── finetuned_parser/ # Trained T5 model (after training)
│   └── regressors/       # RidgeCV diff/imp + XGBoost duration models
├── data/                 # Training datasets
├── tests/                # Parser test suite
├── scripts/              # Visualization and utility scripts
├── docs/
│   ├── backend/          # Backend architecture, API, DB schema, enrichment, scheduling, stats docs
│   └── image_to_prompt/  # Image classification pipeline, data collection, training docs
├── assets/               # Generated charts and visualizations
└── package.json          # Frontend dependencies
```

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- PostgreSQL 15+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

---

### Step 1: Pull Trained Model

```bash
cd VM.AI
python src/parser/pull_from_hf.py
```

This downloads the trained T5 model to `models/finetuned_parser/`.

---

### Step 2: Configure Database

1. Create a PostgreSQL database:

   ```sql
   CREATE DATABASE vmai_db;
   ```

2. Navigate to backend directory:

   ```bash
   cd src/backend
   ```

3. Copy the example environment file:

   ```bash
   copy .env.example .env    # Windows
   # cp .env.example .env    # Linux/Mac
   ```

4. Edit `.env` with your database credentials:
   ```
   DATABASE_URL=postgresql+psycopg://your_user:your_password@localhost:5432/vmai_db
   ```

---

### Step 3: Backend Setup

**Important**: This project uses `uv` - you do NOT need to manually activate virtual environments.

```bash
cd src/backend

# Install dependencies (creates .venv automatically)
uv sync

# Run database migrations
uv run alembic upgrade head

# Start the backend server
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API documentation with interactive testing will be available at: http://127.0.0.1:8000/docs

---

### Step 4: Frontend Setup

```bash
cd VM.AI

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at: http://localhost:5173

---

## Datasets

The T5 parser is trained on three datasets:

### 1. Synthetic Dataset (VMAI_SYNTHETIC_Data.yaml)

- Auto-generated training samples
- Created using template-based data generation
- Contains varied task descriptions with full schema coverage

### 2. Real Dataset (VMAI_REAL_Data.yaml)

- Human-written examples
- More natural language variations
- Includes both "add" and "modify" task patterns

### 3. Specific Dataset (VMAI_SPECIFIC_Data.yaml)

- Targeted examples for fields the model struggles with
- Focused on improving weak areas identified during evaluation

All T5 data follows the pipe-format schema with EXP/PRD tags:

- `[EXP]` - Explicit field (user stated the value directly)
- `[PRD]` - Predicted field (model inferred the value)

Two additional generated datasets support the ML services:

- **VMAI_REGR_Data.csv** — Task text with difficulty/importance labels, used to train the RidgeCV regressors
- **VMAI_DURATION_Data.csv** — Tabular features (difficulty, importance, scheduled duration, category, location, deadline) with real_duration labels, used to train the XGBoost duration model

## Training Pipeline

To retrain the model:

```bash
# From project root
python src/parser/train.py --mode [MODE]
```

Available modes:

- `both` - Mix of add and modify samples (recommended)
- `synthetic` - Only synthetic data
- `real` - Only real human examples
- `specific` - Targeted improvements
- `modify_only` - Modify pattern only (requires existing checkpoint)

Training produces metrics per field (category, difficulty, importance, duration, deadline, location, recurrence).

## Running the Chat Interface

For direct testing without the web interface:

```bash
python src/parser/chat.py
```

Commands:

- `add: <task description>` - Parse a new task
- `modify` - Modify the last added task
- `end` - Exit

## Testing

### Parser Tests (src/parser)

```bash
cd VM.AI

# Core parser functionality
python tests/test_core.py

# Data generation
python tests/test_generator.py

# Add/Modify mode parsing
python tests/test_add.py
python tests/test_modify.py

# Schema conversion
python tests/test_schemas.py

# Dataset validation
python tests/test_validate_dataset.py
python tests/test_data_no_duplicates.py
python tests/test_explicit_fields.py
```

### Backend API Tests (src/backend)

```bash
cd src/backend

# Run all backend tests
uv run pytest

# Individual test modules
uv run pytest tests/test_parser_service.py
uv run pytest tests/test_enrichment.py
uv run pytest tests/test_task_matching.py
uv run pytest tests/test_update_time_score.py
```

## Visualization

Generate charts and graphs for data analysis:

```bash
# Real dataset (7 plots → scripts/output/real/)
python scripts/plot_dataset.py --dataset real

# Specific dataset (7 plots → scripts/output/specific/)
python scripts/plot_dataset.py --dataset specific

# Synthetic dataset (8 plots → scripts/output/synthetic/)
python scripts/plot_synthetic.py

# Training metrics
python scripts/report.py
```

Generated visualizations are saved to `scripts/output/<dataset>/` and can be copied to `assets/` for documentation.

### Real Dataset

![Real Overview](assets/parser/real_overview.png)

### Specific Dataset

![Specific Overview](assets/parser/specific_overview.png)

### Synthetic Dataset

![Synthetic Overview](assets/parser/synthetic_overview.png)

## Documentation

Detailed documentation covering all project components is available in the `docs/` folder:

- `docs/backend/` — Backend architecture, API documentation, database schema, enrichment module, scheduling engine, stats recorder
- `docs/image_to_prompt/` — Image classification pipeline, data collection methodology, training configuration

## Library Versions

Key dependencies:

- Python: 3.12+
- PyTorch: Latest (CUDA recommended for training)
- transformers: Latest (HuggingFace)
- datasets: Latest
- FastAPI: For backend
- SQLAlchemy: 2.0
- React: 19.x (frontend)
- Vite: 8.x (frontend build)

## Limitations and Ethical Considerations

### Data Limitations

- The model is trained on a limited dataset size
- Performance may vary for unusual or ambiguous task descriptions
- Category inference uses task matching (MiniLM cosine similarity) with enrichment fallback; difficulty and importance are predicted through statistics-based overwrites from matched tasks

### Technical Limitations

- The T5-base model has token limits that may affect complex inputs
- The image classifier (EfficientNet-B4) and duration predictor (XGBoost) depend on training data quality
- Time zone handling is not explicitly implemented

### Ethical Considerations

#### Privacy & Data Storage
- Task data is stored in PostgreSQL (single-user demo, no user_id fields)
- Data is NOT encrypted at rest — known demo limitation
- No telemetry, analytics, tracking, or third-party data sharing
- Task data and statistics are used only for enrichment and scheduling
- No personal or confidential data is included in the repository

#### Bias & Fairness
- The T5 parser was fine-tuned on English template data only
- Performance may degrade for non-English input, slang, dialect, or creative phrasing
- The XGBoost regressor was trained on ~1000 labeled examples from a single user — predictions reflect that user's labeling patterns and may not generalize
- No systematic bias analysis has been performed — this is a known limitation

#### Known Risks
- Duration predictions have MAE ≈ 10 minutes — do not rely on them for critical scheduling
- The scheduler has known limitations with overnight tasks (naive datetime, no timezone)
- All ML predictions are estimates; users should verify before committing
- The system is a demo/prototype, not a production scheduling tool

#### Responsible Use
- Always review scheduled tasks before accepting automated changes
- Report unexpected behavior via GitHub Issues
- This is an assistive tool — final scheduling decisions remain with the user

#### Transparency
- Known bugs are documented in `src/backend/logs/`
- Model limitations are discussed in this section
- No deliberate manipulation of results

## Repository Structure

This repository follows best practices for AI projects:

- `/src` - Source code
- `/models` - Trained model files
- `/data` - Training data
- `/docs` - Documentation
- `/assets` - Visualizations and charts
- README.md - Complete project description with setup instructions

No personal or confidential data is included.

## References

- API Documentation: `docs/Frontend_API_Documentation.md`
- HuggingFace Model: vaneaa/vmai-parser
