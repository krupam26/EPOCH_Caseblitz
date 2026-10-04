import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from fastapi.testclient import TestClient
from backend.main import app
import sqlite3
import faiss

client = TestClient(app)

print("=== 1. HEALTH CHECK ===")
h = client.get("/health").json()
print("Health:", h)
assert h["ok"] and h["model_connected"], "Model not connected!"

print("\n=== 2. DATABASE & FAISS SANITY ===")
conn = sqlite3.connect("data/broll.db")
c = conn.cursor()
clips = c.execute("SELECT clip_id, filename, duration, status FROM clips").fetchall()
segs = c.execute("SELECT count(*) FROM segments").fetchone()[0]
print(f"DB clips: {len(clips)}, DB segments: {segs}")
for cl in clips:
    print(" ", cl)
conn.close()

idx = faiss.read_index("data/index/broll.index")
print(f"FAISS vectors count: {idx.ntotal}")
assert idx.ntotal == segs, f"FAISS ({idx.ntotal}) and SQLite ({segs}) count mismatch!"

print("\n=== 3. SEARCH ENDPOINT TEST ===")
res = client.post("/search", json={"query": "girl is walking"}).json()
print("Query: 'girl is walking' -> Results count:", len(res["results"]))
for r in res["results"]:
    print(f"  {r['clip_id']} | {r['percent']}% | {r['start']}s-{r['end']}s | {r['caption']}")

print("\n=== 4. SCRIPT MODE ENDPOINT TEST ===")
script_text = "A girl is walking down the street. The cat is sleeping quietly on a chair. An artist is painting on her easel."
s_res = client.post("/script", json={"text": script_text}).json()
print(f"Script scenes count: {len(s_res['scenes'])}")
for scene in s_res["scenes"]:
    top = scene["results"][0] if scene["results"] else None
    top_info = f"{top['clip_id']} ({top['percent']}%) - {top['caption']}" if top else "No match"
    print(f"  Scene {scene['scene_index']+1}: \"{scene['sentence']}\" -> {top_info}")

print("\n=== 5. STATUS ENDPOINT ===")
st = client.get("/status").json()
print("Status:", st)

print("\n=== ALL SYSTEM CHECKS PASSED PERFECTLY ===")
