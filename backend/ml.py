"""Adapter connecting the FastAPI upload/search flow to CLIP and FAISS."""
from pathlib import Path
from typing import List, Tuple

from ml.clip_model import CLIPModel
from ml.frame_extractor import extract_frames
from ml.segment_builder import build_segments
from ml.vector_store import VectorStore

from .config import INDEX_PATH


_clip = None
_store = None


def _get_models():
    global _clip, _store
    if _clip is None:
        _clip = CLIPModel()
    if _store is None:
        _store = VectorStore()
        index_dir = INDEX_PATH.parent
        metadata_path = index_dir / "metadata.npy"
        if INDEX_PATH.exists() and metadata_path.exists():
            try:
                _store.load(str(index_dir))
            except Exception:
                _store = VectorStore()
    return _clip, _store


def embed_video(path: Path, clip_id: str) -> List[dict]:
    """Embed sampled frames, add averaged segments to FAISS, and save the index."""
    clip, store = _get_models()
    frames = extract_frames(path)
    if not frames:
        raise ValueError("No frames could be extracted from video.")

    for frame in frames:
        frame["embedding"] = clip.encode_image(frame["frame"])

    segments = build_segments(frames)
    if not segments:
        raise ValueError("No video segments could be generated.")

    output = []
    for segment in segments:
        faiss_pos = store.index.ntotal
        meta = {
            "clip_id": clip_id,
            "start": float(segment["start_time"]),
            "end": float(segment["end_time"]),
            "video_path": str(path),
        }
        store.add(segment["embedding"], meta)
        output.append({
            "start": float(segment["start_time"]),
            "end": float(segment["end_time"]),
            "faiss_pos": int(faiss_pos),
        })

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.save(str(INDEX_PATH.parent))
    return output


def search(query: str, k: int) -> List[dict]:
    """Return enriched FAISS search hits with cosine similarity."""
    clip, store = _get_models()
    if store.index.ntotal == 0:
        return []

    from ml.embedding import encode_search_text
    query_vector = encode_search_text(query.strip(), model=clip)
    hits = store.search(query_vector, top_k=k)

    results = []
    for hit in hits:
        pos = int(hit["faiss_id"])
        sim = float(hit["similarity"])
        meta = {}
        if 0 <= pos < len(store.metadata):
            m = store.metadata[pos]
            if isinstance(m, dict):
                meta = m

        results.append({
            "faiss_pos": pos,
            "clip_id": meta.get("clip_id", ""),
            "start_time": meta.get("start", 0.0),
            "end_time": meta.get("end", 0.0),
            "video_path": meta.get("video_path", ""),
            "similarity": sim,
        })
    return results
