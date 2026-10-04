"""FastAPI-facing search adapter."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import TOP_K
from .db import get_conn
from .ml import search as ml_search
from .postprocess import postprocess, split_script
from .schemas import (
    SceneResult,
    ScriptResponse,
    SearchResponse,
    SearchResult,
)
from .thumbnails import get_thumbnail


NO_MATCH_MSG = "No strong match found for this query"
EMPTY_QUERY_MSG = "Please enter a search query"


def _get_frame(result: dict[str, Any]):
    """Load the frame associated with an ML result."""

    thumbnail_path = result.get(
        "thumbnail_path"
    )

    if thumbnail_path:

        path = Path(thumbnail_path)

        if path.exists():

            frame = cv2.imread(
                str(path)
            )

            if frame is not None:
                return frame

    video_path = result.get(
        "video_path"
    )

    start = result.get(
        "start_time",
        result.get("start", 0),
    )

    if not video_path:
        return None

    path = Path(video_path)

    if not path.exists():
        return None

    cap = cv2.VideoCapture(
        str(path)
    )

    try:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            float(start) * 1000,
        )

        ok, frame = cap.read()

        if not ok:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0,
            )

            ok, frame = cap.read()

        return frame if ok else None

    finally:
        cap.release()


def _quality_flag(frame):
    """Return a quality flag for a matched frame."""

    if frame is None:
        return None

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    brightness = float(
        np.mean(gray)
    )

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    if brightness < 35:
        return "dark"

    if blur_score < 30:
        return "blurry"

    return None


def _caption_result(frame):
    """Generate a BLIP caption when available."""

    if frame is None:
        return ""

    try:

        from ml.captioner import get_captioner
        from PIL import Image

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            rgb
        )

        captioner = get_captioner()

        return captioner.caption(
            image
        )

    except Exception:
        return ""


def _similarity_to_percent(
    similarity: float,
):
    """
    Convert similarity to a display percentage.

    CLIP similarity is expected to be in the
    approximate 0-1 range here.
    """

    value = float(
        similarity
    )

    value = max(
        0.0,
        min(1.0, value),
    )

    return int(
        round(value * 100)
    )


def _video_id(
    result: dict[str, Any],
):
    """Return a stable video identifier."""

    video_id = result.get(
        "video_id"
    )

    if video_id:
        return str(video_id)

    clip_id = result.get(
        "clip_id"
    )

    if clip_id:
        return str(clip_id)

    filename = result.get(
        "filename"
    )

    if filename:
        return Path(
            filename
        ).stem

    video_path = result.get(
        "video_path"
    )

    if video_path:
        return Path(
            video_path
        ).stem

    return ""


def _to_postprocess_result(
    result: dict[str, Any],
):
    """
    Convert an ML result into the generic structure
    expected by postprocess.py.
    """

    return {
        "clip_id": _video_id(result),
        "start": float(
            result.get(
                "start_time",
                result.get("start", 0),
            )
        ),
        "end": float(
            result.get(
                "end_time",
                result.get("end", 0),
            )
        ),
        "score": float(
            result.get(
                "similarity",
                result.get("score", 0.0),
            )
        ),
        "_raw": result,
    }


def _normalize_result(result):
    """Accept both legacy FAISS tuples and enriched ML result dictionaries."""
    if isinstance(result, dict):
        if not result.get("video_path") and result.get("clip_id"):
            with get_conn() as connection:
                row = connection.execute(
                    "SELECT c.video_path FROM clips c WHERE c.clip_id=? LIMIT 1",
                    (result["clip_id"],),
                ).fetchone()
                if row:
                    result["video_path"] = row["video_path"]
        return result

    if isinstance(result, (tuple, list)) and len(result) >= 2:
        faiss_pos, similarity = result[0], result[1]
        with get_conn() as connection:
            row = connection.execute(
                "SELECT s.clip_id, s.start_time, s.end_time, c.video_path "
                "FROM segments s JOIN clips c ON c.clip_id = s.clip_id "
                "WHERE s.faiss_pos=? AND c.status='ok' LIMIT 1",
                (int(faiss_pos),),
            ).fetchone()
        if row:
            return {
                "clip_id": row["clip_id"],
                "video_id": row["clip_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "video_path": row["video_path"],
                "similarity": float(similarity),
            }

    return {
        "clip_id": "",
        "start_time": 0.0,
        "end_time": 0.0,
        "similarity": 0.0,
    }


def _to_result(
    result: dict[str, Any],
    percent: int = None,
):
    """Convert an ML result into SearchResult."""

    clip_id = _video_id(
        result
    )

    start = float(
        result.get(
            "start_time",
            result.get("start", 0),
        )
    )

    end = float(
        result.get(
            "end_time",
            result.get("end", start),
        )
    )

    similarity = float(
        result.get(
            "similarity",
            result.get("score", 0.0),
        )
    )

    frame = _get_frame(
        result
    )

    thumbnail_url = (
        f"/thumbnails/"
        f"{clip_id}_{int(start)}.jpg"
    )

    video_url = (
        f"/videos/{clip_id}"
    )

    final_percent = (
        int(percent)
        if percent is not None
        else _similarity_to_percent(similarity)
    )

    return SearchResult(
        clip_id=clip_id,
        start=start,
        end=end,
        percent=final_percent,
        caption=_caption_result(
            frame
        ),
        thumbnail_url=thumbnail_url,
        video_url=video_url,
        quality_flag=_quality_flag(
            frame
        ),
    )


def search_clips(
    query,
    search_fn=None,
    k=TOP_K,
):
    """
    Search the ML index and return the FastAPI response.
    """

    query = (
        query or ""
    ).strip()

    if not query:
        return SearchResponse(
            query=query,
            results=[],
            message=EMPTY_QUERY_MSG,
        )

    try:

        if search_fn is None:
            search_fn = ml_search

        from .config import RAW_K
        raw_results = search_fn(
            query,
            RAW_K,
        )

    except FileNotFoundError:
        raw_results = []

    except Exception as error:
        raise RuntimeError(
            f"B-roll ML search failed: {error}"
        ) from error

    prepared = [
        _to_postprocess_result(
            _normalize_result(result)
        )
        for result in raw_results
    ]

    processed = postprocess(
        prepared,
        k=k,
    )

    converted = []

    for processed_result in processed:

        raw_result = processed_result.get(
            "_raw"
        )
        pct = processed_result.get("percent")

        if raw_result is None:
            raw_result = {
                "video_id": processed_result[
                    "clip_id"
                ],
                "clip_id": processed_result[
                    "clip_id"
                ],
                "start_time": processed_result[
                    "start"
                ],
                "end_time": processed_result[
                    "end"
                ],
                "similarity": processed_result[
                    "score"
                ],
            }

        converted.append(
            _to_result(
                raw_result,
                percent=pct,
            )
        )

    return SearchResponse(
        query=query,
        results=converted,
        message=(
            None
            if converted
            else NO_MATCH_MSG
        ),
    )


def search_script(
    text,
    search_fn=None,
    k=TOP_K,
):
    """Search the ML index for every sentence in a script."""

    sentences, truncated = split_script(
        text
    )

    scenes = []

    for index, sentence in enumerate(
        sentences,
    ):

        response = search_clips(
            sentence,
            search_fn=search_fn,
            k=k,
        )

        scenes.append(
            SceneResult(
                scene_index=index,
                sentence=sentence,
                results=response.results,
            )
        )

    return ScriptResponse(
        scenes=scenes,
        truncated=truncated,
    )


class BrollSearch:
    """Compatibility wrapper for existing callers."""

    def search(
        self,
        query,
        top_k=TOP_K,
    ):
        return ml_search(
            query,
            top_k,
        )