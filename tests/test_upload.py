import zipfile
import cv2
import numpy as np
import pytest

from backend import config, db, upload
from backend.zip_utils import UploadError


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(upload, "VIDEO_DIR", tmp_path / "videos")


def make_video(path, seconds=3, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for _ in range(seconds * fps):
        w.write(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    w.release()


def test_good_and_corrupt(tmp_path):
    make_video(tmp_path / "good.mp4")
    zp = tmp_path / "lib.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(tmp_path / "good.mp4", "good.mp4")
        z.writestr("broken.mp4", b"not a video")
    calls = []
    res = upload.process_upload(zp, progress_cb=lambda i, n, f: calls.append((i, n)))
    assert res.clips_indexed == 1
    assert res.skipped == ["broken.mp4"]
    assert res.segments_indexed >= 1
    assert calls[-1] == (2, 2)


def test_bad_zip(tmp_path):
    p = tmp_path / "x.zip"
    p.write_bytes(b"nope")
    with pytest.raises(UploadError):
        upload.process_upload(p)


def test_video_longer_than_60s(tmp_path, monkeypatch):
    # Mock probe_video or test duration guard
    make_video(tmp_path / "long.mp4", seconds=2, fps=10)
    zp = tmp_path / "long.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(tmp_path / "long.mp4", "long.mp4")

    # Simulate video duration probe returning 65.0s
    monkeypatch.setattr(upload, "probe_video", lambda p: (_ for _ in ()).throw(ValueError("Clip duration (65.0s) exceeds maximum limit of 60s")))
    res = upload.process_upload(zp)
    assert "long.mp4" in res.skipped
    assert res.clips_indexed == 0


def test_zip_larger_than_500mb(tmp_path, monkeypatch):
    p = tmp_path / "huge.zip"
    p.write_bytes(b"dummy")
    # Simulate stat size exceeding MAX_ZIP_BYTES
    from unittest.mock import MagicMock
    mock_stat = MagicMock()
    mock_stat.st_size = 501 * 1024 * 1024
    monkeypatch.setattr(Path, "stat", lambda self: mock_stat)
    with pytest.raises(UploadError, match="larger than 500 MB"):
        upload.process_upload(p)
