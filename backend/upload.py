"""Upload handler for B-roll ZIP files."""

import logging
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from .config import (
    MAX_CLIP_SECONDS,
    VIDEO_DIR,
    VIDEO_EXTS,
)
from .db import get_conn, init_db
from .schemas import UploadResponse
from .zip_utils import safe_extract, validate_zip


log = logging.getLogger(__name__)


EmbedFn = Callable[
    [Path, str],
    List[Dict],
]


def probe_video(
    path: Path,
) -> float:
    """
    Return video duration in seconds.

    Raises ValueError when the video cannot be opened
    or decoded.
    """

    cap = cv2.VideoCapture(
        str(path)
    )

    try:

        if not cap.isOpened():
            raise ValueError(
                "cannot open video file"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frames = cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        ok, _ = cap.read()

        if (
            not ok
            or not fps
            or fps <= 0
            or frames <= 0
        ):
            raise ValueError(
                "cannot decode video frames"
            )

        duration = float(frames / fps)
        if duration > MAX_CLIP_SECONDS:
            raise ValueError(
                f"Clip duration ({duration:.1f}s) exceeds maximum limit of {MAX_CLIP_SECONDS}s"
            )
        return duration

    finally:
        cap.release()


def stub_embed(
    path: Path,
    clip_id: str,
    duration: float = 0.0,
) -> List[Dict]:
    """
    Temporary fallback when the ML model is unavailable.

    Creates approximately 6-second segments without embeddings.
    """

    end = min(
        duration,
        MAX_CLIP_SECONDS,
    )

    segments = []

    t = 0.0

    while t < end:

        segments.append(
            {
                "start": t,
                "end": min(
                    t + 6.0,
                    end,
                ),
                "faiss_pos": None,
            }
        )

        t += 6.0

    return segments


def process_upload(
    zip_path: Path,
    embed_fn: Optional[EmbedFn] = None,
    progress_cb: Optional[
        Callable[[int, int, str], None]
    ] = None,
) -> UploadResponse:
    """
    Validate ZIP -> extract -> process videos -> index videos
    -> store metadata in SQLite.
    """

    init_db()

    zip_path = Path(
        zip_path
    )

    validate_zip(
        zip_path
    )

    job_id = uuid.uuid4().hex[:8]

    dest = (
        VIDEO_DIR
        / job_id
    )

    safe_extract(
        zip_path,
        dest,
    )

    videos = sorted(
        p
        for p in dest.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in VIDEO_EXTS
            and "__MACOSX"
            not in p.parts
        )
    )

    with get_conn() as c:

        c.execute(
            """
            INSERT INTO jobs
                (job_id, total, processed, status)
            VALUES
                (?, ?, 0, 'processing')
            """,
            (
                job_id,
                len(videos),
            ),
        )

    skipped: List[str] = []

    clips_done = 0
    segs_done = 0

    for i, video in enumerate(
        videos,
        start=1,
    ):

        clip_id = (
            f"{job_id}_{i:03d}"
        )

        try:

            duration = probe_video(
                video
            )

            if duration <= 0:
                raise ValueError(
                    "video has invalid duration"
                )

            if embed_fn:

                segments = embed_fn(
                    video,
                    clip_id,
                )

            else:

                segments = stub_embed(
                    video,
                    clip_id,
                    duration,
                )

            with get_conn() as c:

                c.execute(
                    """
                    INSERT INTO clips
                        (
                            clip_id,
                            job_id,
                            filename,
                            video_path,
                            duration,
                            status
                        )
                    VALUES
                        (?, ?, ?, ?, ?, 'ok')
                    """,
                    (
                        clip_id,
                        job_id,
                        video.name,
                        str(video),
                        duration,
                    ),
                )

                if segments:

                    c.executemany(
                        """
                        INSERT INTO segments
                            (
                                clip_id,
                                start_time,
                                end_time,
                                faiss_pos
                            )
                        VALUES
                            (?, ?, ?, ?)
                        """,
                        [
                            (
                                clip_id,
                                float(
                                    segment[
                                        "start"
                                    ]
                                ),
                                float(
                                    segment[
                                        "end"
                                    ]
                                ),
                                segment.get(
                                    "faiss_pos"
                                ),
                            )
                            for segment
                            in segments
                        ],
                    )

            clips_done += 1
            segs_done += len(
                segments
            )

        except Exception as e:

            log.exception(
                "Skipping video %s",
                video.name,
            )

            skipped.append(
                video.name
            )

            with get_conn() as c:

                c.execute(
                    """
                    INSERT INTO clips
                        (
                            clip_id,
                            job_id,
                            filename,
                            video_path,
                            status,
                            error
                        )
                    VALUES
                        (?, ?, ?, ?, 'skipped', ?)
                    """,
                    (
                        clip_id,
                        job_id,
                        video.name,
                        str(video),
                        str(e),
                    ),
                )

        with get_conn() as c:

            c.execute(
                """
                UPDATE jobs
                SET processed=?
                WHERE job_id=?
                """,
                (
                    i,
                    job_id,
                ),
            )

        if progress_cb:

            progress_cb(
                i,
                len(videos),
                video.name,
            )

    with get_conn() as c:

        c.execute(
            """
            UPDATE jobs
            SET status='done'
            WHERE job_id=?
            """,
            (job_id,),
        )

    return UploadResponse(
        job_id=job_id,
        clips_indexed=clips_done,
        segments_indexed=segs_done,
        skipped=skipped,
    )