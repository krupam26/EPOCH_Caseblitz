"""Build timestamped segment embeddings from sampled frames."""
import numpy as np


def build_segments(frames, group_size=3, frame_interval=2.0, duration=None, stride=None):
    segments = []
    step = stride if stride is not None and stride > 0 else group_size
    for index in range(0, len(frames), step):
        group = frames[index:index + group_size]
        if not group:
            continue

        embeddings = np.asarray([item["embedding"] for item in group], dtype=np.float32)
        average = embeddings.mean(axis=0)
        average /= np.linalg.norm(average) + 1e-12
        segments.append({
            "start_time": float(group[0]["timestamp"]),
            "end_time": min(
                float(group[-1]["timestamp"]) + frame_interval,
                duration if duration is not None else float("inf"),
            ),
            "embedding": average,
        })
    return segments
