from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"          # /data/videos/{job_id}/
DB_PATH = DATA_DIR / "broll.db"
INDEX_PATH = DATA_DIR / "index" / "broll.index"  # FAISS index file

MAX_ZIP_BYTES = 500 * 1024 * 1024   # 500 MB
MAX_CLIPS = 50
MAX_CLIP_SECONDS = 60
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Search post-processing (calibrated for open_clip ViT-B-32)
MIN_SCORE, MAX_SCORE = 0.12, 0.30
PERCENT_THRESHOLD = 25
MAX_SEGMENTS_PER_CLIP = 2
MIN_WORDS = 3
MAX_SENTENCES = 20

# Search
RAW_K = 20     # raw hits fetched from FAISS before dedup/threshold
TOP_K = 5      # results returned to the UI

THUMB_DIR = DATA_DIR / "thumbnails"
