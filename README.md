# Video Summarization API

A FastAPI service that accepts a video upload and produces a concise summarized
video with AI-generated narration.

## Architecture

```
video upload
    │
    ▼
┌──────────────┐    ┌───────────────┐    ┌─────────┐    ┌────────────┐
│  WhisperX    │──▶│  Ollama LLM   │──▶│  gTTS /  │──▶│  MoviePy   │
│  Transcribe  │    │  Summarize    │    │  FPT TTS │    │  Compose   │
└──────────────┘    └───────────────┘    └─────────┘    └────────────┘
                                                              │
                                                              ▼
                                                     summarized video
```

**Design Patterns**:
- **Strategy** — swappable transcription, summarization, and TTS backends
- **Factory** — `ComponentFactory` creates the correct adapter from config
- **Repository** — `TaskRepository` abstracts PostgreSQL persistence
- **Pipeline** — `VideoSummarizationPipeline` orchestrates the workflow

## Prerequisites

- Python 3.11+
- FFmpeg installed and on `PATH`
- **PostgreSQL 14+** running locally (or via Docker)
- Ollama running locally (for LLM summarization)
- NVIDIA GPU + CUDA (optional — auto-falls back to CPU)

## Quick Start

```bash
# 1. Clone and enter
git clone <repo-url>
cd video-summarization-api

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/macOS
# Edit .env with your DATABASE_URL and other settings

# 5. Set up the database
# Option A: Let the app auto-create tables on first start
python main.py

# Option B: Use Alembic migrations (recommended for production)
alembic upgrade head
python main.py
```

## Database Setup

### Using Docker (easiest)

```bash
docker run -d \
  --name vsapi-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=video_summarization \
  -p 5432:5432 \
  postgres:16-alpine
```

### Using a local PostgreSQL installation

```sql
CREATE DATABASE video_summarization;
```

Then update `DATABASE_URL` in your `.env` file.

### Alembic Migrations

```bash
# Apply all migrations
alembic upgrade head

# Create a new migration after model changes
alembic revision --autogenerate -m "describe your change"

# Roll back the last migration
alembic downgrade -1
```

The API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (includes DB status) |
| `POST` | `/api/v1/summarize` | Upload video & start pipeline |
| `GET` | `/api/v1/tasks/{id}` | Check task status |
| `GET` | `/api/v1/tasks/{id}/download` | Download result |

### Upload Example

```bash
curl -X POST http://localhost:8000/api/v1/summarize \
  -F "file=@my_video.mp4" \
  -G -d "language=vi" -d "tts_env=local"
```

### Check Status

```bash
curl http://localhost:8000/api/v1/tasks/<task-id>
```

### Download Result

```bash
curl -O http://localhost:8000/api/v1/tasks/<task-id>/download
```

## Configuration

All settings are read from environment variables (or `.env` file).
See [`.env.example`](.env.example) for the full list.

## Project Structure

```
├── main.py                    # Entry point
├── requirements.txt
├── alembic.ini                # Alembic configuration
├── .env.example
├── alembic/
│   ├── env.py                 # Async migration environment
│   ├── script.py.mako         # Migration template
│   └── versions/
│       └── 001_create_tasks_table.py
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints + lifespan
│   ├── core/
│   │   ├── config.py          # Pydantic settings (incl. DATABASE_URL)
│   │   └── logging.py         # Logging setup
│   ├── db/
│   │   ├── models.py          # SQLAlchemy ORM models
│   │   └── session.py         # Async engine & session factory
│   ├── models/
│   │   └── schemas.py         # Pydantic request / response models
│   ├── patterns/
│   │   ├── interfaces.py      # Abstract strategies (ABC)
│   │   ├── adapters.py        # Concrete implementations
│   │   └── factory.py         # Component factory
│   └── services/
│       ├── pipeline.py        # Orchestration pipeline
│       └── task_store.py      # TaskRepository (PostgreSQL)
├── uploads/                   # Uploaded videos (gitignored)
└── outputs/                   # Summarized videos (gitignored)
```

## License

MIT
