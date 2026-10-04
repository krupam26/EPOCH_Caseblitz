"""Fixed-interval video frame extraction with blurry frame enhancement for CLIP indexing."""
import cv2
import numpy as np

MAX_VIDEO_DURATION = 60.0


def enhance_if_blurry(rgb_frame, blur_threshold=80.0):
    """
    Detect if frame is blurry (Laplacian variance < threshold)
    and apply CLAHE and unsharp masking to extract visual context.
    """
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if blur_score < blur_threshold:
        # 1. CLAHE adaptive contrast enhancement on luminance channel
        lab = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

        # 2. Unsharp masking to recover edge and contour details
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        return sharpened, True, blur_score

    return rgb_frame, False, blur_score


def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    duration = total_frames / fps
    if duration > MAX_VIDEO_DURATION:
        cap.release()
        return []

    interval = 2 if duration < 30 else 4
    timestamps = list(range(0, max(1, int(duration)), interval))[:15]
    frames = []

    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = cap.read()
        if success:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enhanced_rgb, was_blurry, blur_score = enhance_if_blurry(rgb)
            frames.append({
                "timestamp": float(timestamp),
                "frame": enhanced_rgb,
                "raw_frame": rgb,
                "is_blurry": was_blurry,
                "blur_score": blur_score,
            })

    cap.release()
    return frames
