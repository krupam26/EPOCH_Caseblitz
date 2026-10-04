"""Command-line entry point for indexing and searching b-roll."""
import argparse
from pathlib import Path

from ml.indexing import rebuild_library
from ml.search import search_videos


ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "data" / "videos"


def main():
    parser = argparse.ArgumentParser(description="B-roll ML pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index video files")
    index_parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Video paths; defaults to data/videos/*.mp4",
    )

    search_parser = subparsers.add_parser("search", help="Search the saved index")
    search_parser.add_argument("query")
    search_parser.add_argument("--threshold", type=float, default=0.25)
    search_parser.add_argument("--top-k", type=int, default=5)

    arguments = parser.parse_args()
    if arguments.command == "index":
        paths = arguments.paths or sorted(VIDEO_DIR.glob("*.mp4"))
        if not paths:
            parser.error("No videos found to index")
        segments = rebuild_library(paths)
        print(f"Indexed {len(paths)} videos into {len(segments)} segments.")
        return

    results = search_videos(
        arguments.query,
        threshold=arguments.threshold,
        top_k=arguments.top_k,
    )
    if not results:
        print("No relevant footage found.")
        return
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['filename']} | "
            f"{result['start_time']:.1f}s - {result['end_time']:.1f}s | "
            f"confidence {result['confidence_score']}%"
        )


if __name__ == "__main__":
    main()
