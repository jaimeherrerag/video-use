"""Print word-level timestamps from a Scribe transcript in a given range."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True, type=Path)
    ap.add_argument("--start", required=True, type=float)
    ap.add_argument("--end", required=True, type=float)
    args = ap.parse_args()

    data = json.loads(args.transcript.read_text(encoding="utf-8"))
    for w in data.get("words", []):
        if w.get("type") != "word":
            continue
        ws = w.get("start"); we = w.get("end")
        if ws is None or we is None:
            continue
        if ws < args.start or ws > args.end:
            continue
        text = (w.get("text") or "").strip()
        print(f"{ws:7.2f}-{we:7.2f}  {text!r}")


if __name__ == "__main__":
    main()
