"""Saved-index semantic search for text and scripts."""
import re
from ml.embedding import encode_search_text
from ml.ranking import rank_results
from ml.storage import get_all_segment_metadata, load_index

MAX_QUERY_WORDS = 500


def validate_query(query):
    cleaned = (query or "").strip()
    if len(cleaned.split()) > MAX_QUERY_WORDS:
        raise ValueError("Search input cannot exceed 500 words")
    return cleaned


def _extract_activity(query):
    match = re.search(r"\b(?:is|are|was|were)\s+(?:doing\s+)?(.+)$", query, flags=re.IGNORECASE)
    if not match:
        return None
    return re.sub(r"[.!?,]+$", "", match.group(1)).strip() or None


def search_videos(
    query,
    top_k=5,
    threshold=0.20,
    max_per_video=1,
    model=None,
    rerank_frames=True,
    frame_interval=2.0,
    frames_per_segment=2,
    segment_stride=None,
    use_captions=False,
):
    """Search the saved FAISS index and return ranked timestamped results."""
    query = validate_query(query)
    if not query:
        return []

    try:
        index = load_index()
    except FileNotFoundError:
        return []
    if index.ntotal == 0:
        return []

    query_vector = encode_search_text(query.strip(), model=model)
    activity = _extract_activity(query)
    if activity and activity.lower() != query.strip().lower():
        import numpy as np
        activity_vector = encode_search_text(activity, model=model)
        blended = 0.82 * query_vector + 0.18 * activity_vector
        query_vector = blended / (np.linalg.norm(blended) + 1e-12)

    scores, ids = index.search(query_vector.reshape(1, -1), index.ntotal)
    metadata = {
        row["faiss_id"]: row for row in get_all_segment_metadata()
    }
    results = []
    counts = {}

    for score, faiss_id in zip(scores[0], ids[0]):
        if faiss_id < 0 or float(score) < threshold:
            break
        row = metadata.get(int(faiss_id))
        if row is None:
            continue
        video_key = row["video_path"]
        if counts.get(video_key, 0) >= max_per_video:
            continue
        results.append({
            **row,
            "similarity": float(score),
            "explanation": (
                "Matched using text-to-visual embedding similarity."
            ),
        })
        counts[video_key] = counts.get(video_key, 0) + 1
        if len(results) >= top_k:
            break

    return rank_results(results, threshold=0, max_results=top_k)


class BrollSearch:
    def __init__(self, model=None):
        self.model = model

    def search(self, query, top_k=20):
        return search_videos(query, top_k=top_k, model=self.model)
