"""Re-transcribe blob regions (Scribe collapsed them to single mega-words with
no internal word timestamps) as short clips, offset their timestamps back to
the master timeline, and merge with the good words outside those regions.
Output: edit/transcripts/words_corrected.json
"""
import json, subprocess, tempfile, sys
from pathlib import Path
from transcribe import load_api_key, call_scribe

SRC = r"C:/Users/Jaime Herrera/Videos/youtube/36 (01-06-26)/2026-06-01 21-51-41.mp4"
EDIT = Path(r"C:/dev/Video editing/projects/2026-06-01_video-36/edit")
ORIG = EDIT / "transcripts" / "2026-06-01 21-51-41.json"

# (start, end, label) — regions to re-transcribe at clip-level
REGIONS = [
    (100.0, 135.0, "r_llevamos"),
    (350.0, 460.0, "r_openrouter"),
    (655.0, 760.0, "r_mas_dicho"),
    (1390.0, 1900.0, "r_big"),
    (2208.0, 2236.0, "r_carpeta"),
]

def extract(start, end, dest):
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", SRC,
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    key = load_api_key()
    orig = json.load(open(ORIG, encoding="utf-8"))
    words = [w for w in orig["words"] if w["type"] == "word" and w["text"].strip()]

    cache = EDIT / "transcripts" / "_blob_retx_cache.json"
    if cache.exists():
        retx = json.load(open(cache, encoding="utf-8"))
    else:
        retx = {}
        with tempfile.TemporaryDirectory() as tmp:
            for (s, e, label) in REGIONS:
                wav = Path(tmp) / f"{label}.wav"
                print(f"  extracting {label} {s}-{e}", flush=True)
                extract(s, e, wav)
                mb = wav.stat().st_size / 1e6
                print(f"  scribe {label} ({mb:.1f} MB)...", flush=True)
                resp = call_scribe(wav, key, language="es")
                ws = [{"text": w["text"], "start": round(w["start"] + s, 3),
                       "end": round(w["end"] + s, 3)}
                      for w in resp["words"] if w["type"] == "word" and w["text"].strip()]
                retx[label] = {"start": s, "end": e, "words": ws}
                print(f"    -> {len(ws)} words", flush=True)
        json.dump(retx, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # build corrected: drop original words whose start falls in any region, add retx
    covered = [(r["start"], r["end"]) for r in retx.values()]
    def in_region(t):
        return any(s <= t < e for (s, e) in covered)
    kept = [{"text": w["text"], "start": w["start"], "end": w["end"]}
            for w in words if not in_region(w["start"])]
    for r in retx.values():
        kept.extend(r["words"])
    kept.sort(key=lambda w: w["start"])
    out = EDIT / "transcripts" / "words_corrected.json"
    json.dump(kept, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"corrected words: {len(kept)} -> {out}")

if __name__ == "__main__":
    main()
