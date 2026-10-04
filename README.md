# FrameFind — AI-Powered B-Roll Search & Narrative Retrieval Engine

> **A narrative-aware B-roll retrieval system:** Paste a script or narration, not just keywords, and get scene-by-scene ranked clip suggestions with timestamps, explainability captions, and confidence-based filtering.

FrameFind solves the tedious process of video editors manually scrubbing through hours of un-tagged footage. Using **OpenCLIP ViT-B-32 visual embeddings**, **FAISS vector indexing**, and **BLIP explainability captioning**, FrameFind matches footage by semantic and visual meaning.

---

## Team

| Name | Role | Responsibilities |
| :--- | :--- | :--- |
| **Siya Rozani** | Frontend & UI Integration | React + TypeScript UI, unified neon console, video modal player with timestamp seeking, search & script UX |
| **Krupa Mehta** | Machine Learning & Retrieval | OpenCLIP embeddings, prompt ensembling, FAISS indexing, BLIP captioning, blurry frame pre-processing |
| **Kavya Chauhan** | Backend & API Architecture | FastAPI backend, SQLite metadata/segment store, ZIP validation, duration/corruption guards, post-processing |

---

## Key Features & USP

### 1. Script Mode (Our Headline USP)
Most search tools only match single keywords. In FrameFind's **Script Mode**, editors paste an entire script or creative brief:
- The backend splits narration into distinct scenes/sentences.
- Each scene is queried against the indexed video library in parallel.
- Returns a **scene-by-scene breakdown** with timestamps, percentage match scores, and captions.

### 2. Semantic Natural-Language Search
Search footage using natural descriptive queries (e.g. *"a woman walking outdoors"*, *"a cat resting on a chair"*, *"artist painting on an easel"*). Queries use CLIP prompt-ensembling templates to guarantee robust zero-shot visual retrieval.

### 3. Timestamp Localization
For any video clip, FrameFind pinpoints the exact **segment range** (e.g., `0:00–0:06` or `0:12–0:22`) where the visual match occurs. The web player jumps playback straight to that timestamp.

### 4. Calibrated Match Percentage (0–100%)
Raw cosine similarity (`~0.15–0.35`) is rescaled via calibrated min-max mapping into an intuitive, user-friendly percentage score (`0–100%`) rather than confusing raw decimals.

### 5. AI Explainability Captions
Top search matches feature on-demand **BLIP image captions** (e.g. *"a woman walking down a stone walkway next to a canal"*) so the editor immediately understands *why* that footage was retrieved.

### 6. Blurry Frame Context Extraction
Blurry video frames (`cv2.Laplacian` variance < 80.0) undergo OpenCV pre-processing:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** on the luminance channel (LAB color space) to reveal hidden textures.
- **Unsharp Masking** to enhance edges and contours before embedding generation.

### 7. Strict Input & Quality Constraints
- **Max ZIP Archive:** 500 MB hard cap (flagged immediately).
- **Max Video Clips:** 50 clips per archive (flagged immediately).
- **Max Clip Duration:** 60 seconds (longer clips flagged immediately as skipped).
- **Corrupted Videos:** OpenCV decode validation skips unreadable files without crashing the upload batch.
- **No-Match Handling:** Queries scoring below threshold clearly show *"No strong match found for this query"*.
- **Deduplication:** Caps results at 2 segments per clip to prevent a single clip from dominating.

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                       USER BROWSER                          │
│         React + TypeScript + Vite UI (FrameFind Console)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
            REST API Calls (Vite Proxy in Dev)
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND (run.py)                 │
│  /upload          /search          /script          /health │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
    ┌──────────────────────┐        ┌──────────────────────┐
    │  SQLite (broll.db)   │        │     ML Pipeline      │
    │  - jobs              │        │  - OpenCLIP ViT-B-32 │
    │  - clips (status/err)│        │  - FAISS IndexFlatIP │
    │  - segments (timing) │        │  - BLIP Captioner    │
    └──────────────────────┘        │  - OpenCV Enhancement│
                                    └──────────────────────┘
```

---

## Directory Structure

```
EPOCH_Caseblitz/
├── backend/                  # FastAPI backend
│   ├── config.py             # Thresholds, paths, hard constraints
│   ├── db.py                 # SQLite schema, tables (jobs, clips, segments)
│   ├── main.py               # FastAPI application routes
│   ├── ml.py                 # Backend adapter to CLIP & FAISS
│   ├── postprocess.py        # Percentage rescaling, deduplication, thresholds
│   ├── schemas.py            # Pydantic request/response models
│   ├── search.py             # Search orchestration & BLIP captioning
│   ├── thumbnails.py         # Frame thumbnail extraction & caching
│   ├── upload.py             # ZIP ingestion, duration checks, SQLite indexing
│   └── zip_utils.py          # ZIP validation (500MB cap, 50-clip cap, zip-slip defense)
├── frontend/                 # React + TypeScript + Vite UI
│   ├── src/
│   │   ├── components/       # LandingPage, WorkspaceShell, UploadPanel, VideoModal
│   │   ├── api.ts            # Typed client for backend REST API
│   │   ├── App.tsx           # State management & view routing
│   │   └── index.css         # Dark theme & acid-green design system
│   ├── package.json
│   └── vite.config.ts        # Vite proxy configuration
├── ml/                       # Core Machine Learning modules
│   ├── captioner.py          # BLIP captioning model
│   ├── clip_model.py         # OpenCLIP ViT-B-32 model wrapper
│   ├── embedding.py          # Prompt ensembling & image encoding
│   ├── frame_extractor.py    # Frame sampling & CLAHE blur enhancement
│   ├── indexing.py           # Video library indexing pipeline
│   ├── ranking.py            # Ranking & percentage rescaling
│   ├── script_processor.py   # Sentence splitting & script scene search
│   ├── search.py             # Core FAISS search logic & vector blending
│   ├── segment_builder.py    # Segment grouping with stride support
│   └── vector_store.py       # FAISS IndexFlatIP vector database
├── docs/                     # Documentation & specifications
│   ├── ARCHITECTURE.md
│   ├── DEMO.md
│   ├── ML_GUIDE.md
│   ├── broll_9hr_work_division.pdf
│   └── context.md            # Single source of truth & hackathon scope
├── scripts/                  # Management scripts
│   ├── check_integration.py
│   ├── reset_data.py
│   └── verify_system.py      # Full end-to-end endpoint & model verification
├── test_videos/              # Sample test footage (video1..video5)
├── tests/                    # Comprehensive unit & integration tests
│   ├── test_api.py
│   ├── test_ml_components.py # Tests for normalization, blur enhancement, quality
│   ├── test_pipeline.py      # Pipeline and stride tests
│   ├── test_postprocess.py   # Rescaling & deduplication tests
│   ├── test_search.py        # Search ranking tests
│   ├── test_upload.py        # 500MB, 60s duration, corruption tests
│   └── test_zip_utils.py     # ZIP limits & zip-slip tests
├── tools/                    # Benchmarks & CLI search tools
│   ├── cli_search.py         # CLI runner for indexing and search
│   ├── benchmark_ingestion.py
│   ├── evaluate_configurations.py
│   ├── evaluate_search.py
│   └── prepare_dataset.py
├── run.py                    # Root FastAPI launcher
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Quick Start & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ and `pnpm` (or `npm`)

### 1. Clone & Set Up Python Environment
```bash
git clone https://github.com/krupam26/EPOCH_Caseblitz.git
cd EPOCH_Caseblitz

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the FastAPI Backend
```bash
python run.py
```
- API runs at: `http://127.0.0.1:8000`
- Interactive Swagger docs: `http://127.0.0.1:8000/docs`

### 3. Run the Frontend
In a separate terminal:
```bash
cd frontend
pnpm install
pnpm dev
```
- Open browser at: `http://localhost:5173`

---

## Configuration & Environment Variables (`.env`)

The repository includes a default `.env` file:
```bash
VITE_API_BASE_URL=
```
- **Why is it empty?** FrameFind is designed as an **open-access, zero-auth system** for seamless hackathon evaluation. In local development, the Vite dev server automatically proxies all API calls (`/search`, `/upload`, `/script`, `/videos`, `/thumbnails`) directly to `http://127.0.0.1:8000`. Leaving `VITE_API_BASE_URL` blank ensures the proxy handles all requests without needing CORS or IP configuration.
- **For Production Deployment:** Set `VITE_API_BASE_URL=https://your-backend-domain.com`.

---

## Running the Automated Test Suite

Run all unit and integration tests across ML, backend, and upload constraints:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
All 22 unit tests validate:
- ZIP file limit <= 500 MB & clip count <= 50
- Video clip duration guard <= 60s
- Corrupted video handling and skipped logging
- Blurry frame enhancement (CLAHE + unsharp masking)
- Stride-based segment building
- Score percentage calibration and clip deduplication
- REST API `/health`, `/search`, and `/script` contracts

---

## Sample Queries

| Query | Expected Result | Top Match in Demo Data |
| :--- | :--- | :--- |
| `"girl is walking"` | Woman walking along stone path | `video2.mp4` (~51%) |
| `"cat"` | Cat resting on pink chair | `video1.mp4` (100%) |
| `"girl painting"` | Artist painting at an easel | `video3.mp4` (100%) |

---

## Scope & Design Choices

| Feature | Decision | Rationale |
| :--- | :--- | :--- |
| **Authentication** | Explicitly Omitted | Open-access demo for judges with zero friction. |
| **Scene Detection** | Fixed Time Windows | Fixed-interval sampling (2–4s) balances accuracy with fast CPU/GPU inference. |
| **Explainability** | Query-Time BLIP | Captions generated only for top results to ensure sub-second indexing. |
| **Storage** | Local SQLite + FAISS | Eliminates cloud dependencies for maximum reliability during live judging. |
