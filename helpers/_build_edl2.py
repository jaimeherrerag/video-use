"""Build edl.json v2 — feedback round 1.
- grade none
- drop "es decir" restatement + "why free / comparison teaser" section (keep CTA)
- general trailing-silence trim (snap each segment end to real speech end)
- insert re-recorded clip (2026-06-02) after the "settings json listo" segment
"""
import json, re
from pathlib import Path

base = Path(r"C:/dev/Video editing/projects/2026-06-01_video-36/edit")
SRC = r"C:/Users/Jaime Herrera/Videos/youtube/36 (01-06-26)/2026-06-01 21-51-41.mp4"
NEW = r"C:/Users/Jaime Herrera/Videos/youtube/36 (01-06-26)/2026-06-02 15-17-17.mp4"
SRC_KEY = "2026-06-01 21-51-41"
NEW_KEY = "2026-06-02 15-17-17"
HEAD, TAIL = 0.05, 0.08

runs = json.load(open(base / "runs.json", encoding="utf-8"))

def parse_sil(path):
    txt = path.read_text(encoding="utf-8")
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", txt)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", txt)]
    return list(zip(starts, ends))

sil_main = parse_sil(base / "sil_main.txt")
sil_new = parse_sil(base / "sil_new.txt")

def trim_trailing(end, sil):
    """If `end` falls inside (or just past) a silence interval, snap to its start."""
    for ss, se in sil:
        if ss < end <= se + 0.05:
            return round(ss + 0.06, 3)
    return end

def trim_leading(start, sil):
    """If `start` falls inside a silence interval, snap forward to speech onset."""
    for ss, se in sil:
        if ss - 0.05 <= start < se - 0.05:
            return round(se - 0.04, 3)
    return start

DROP = {
    49.06, 71.94,
    285.58, 290.52,
    255.22,                            # #5 "dentro de este auto. Es decir." (reformulacion)
    530.16, 533.83,
    561.11, 582.87,
    647.59, 656.96,
    936.07, 942.49, 951.79,
    1021.15, 1029.03, 1042.01, 1043.09,
    1141.78,
    1198.54, 1208.30,
    1229.38, 1236.12,
    1493.46,
    1611.10, 1619.56,
    1782.77, 1798.86,
    2074.67,
    1911.38,
    1969.98, 1971.82,
    2218.14,
    2514.70, 2519.55,
    745.90,
    # #6 seccion "por que son gratuitos" + "mas adelante comparaciones" (conservar CTA)
    491.83, 516.05, 537.37, 584.51, 596.37, 659.82, 677.40, 684.74, 691.88,
}

TRIM = {
    112.28: {"end": 115.00},
    310.22: {"end": 313.10},
    413.96: {"end": 419.90},
    448.32: {"end": 454.55},
    1064.59: {"end": 1067.05},
    1485.78: {"end": 1490.63},
    1724.34: {"start": 1742.90},
    1777.35: {"end": 1778.85},
    1793.01: {"end": 1795.52},
    2574.38: {"end": 2576.18},
    2579.49: {"start": 2580.58},
}

def build_ranges(runs_list, src_key, sil, default_head=HEAD, default_tail=TAIL, drop=None, trim=None):
    drop = drop or set(); trim = trim or {}
    out = []
    for r in runs_list:
        st = r["s"]
        if st in drop:
            continue
        if not r["text"].strip(" .").strip():
            continue
        start = r["s"] - default_head
        end = r["e"] + default_tail
        if st in trim:
            ov = trim[st]
            if "start" in ov: start = ov["start"]
            if "end" in ov: end = ov["end"]
        if not (st in trim and "start" in trim[st]):
            start = trim_leading(start, sil)     # snap out of leading silence
        end = trim_trailing(end, sil)            # snap out of trailing silence
        out.append({"source": src_key, "start": round(start, 3), "end": round(end, 3),
                    "quote": r["text"][:90]})
    return out

main_ranges = build_ranges(runs, SRC_KEY, sil_main, drop=DROP, trim=TRIM)

# --- new clip runs (gap 0.8) ---
nd = json.load(open(base / "transcripts" / f"{NEW_KEY}.json", encoding="utf-8"))
nw = [w for w in nd["words"] if w["type"] == "word" and w["text"].strip()]
nruns = []; cur = [nw[0]]
for w in nw[1:]:
    if w["start"] - cur[-1]["end"] > 0.8: nruns.append(cur); cur = [w]
    else: cur.append(w)
nruns.append(cur)
nrun_objs = [{"s": round(r[0]["start"], 2), "e": round(r[-1]["end"], 2),
              "text": "".join(" " + x["text"] for x in r).strip()} for r in nruns]
new_ranges = build_ranges(nrun_objs, NEW_KEY, sil_new)

# --- insert new clip after "settings json listo" (src ~1675.5) ---
ins = next(i for i, r in enumerate(main_ranges) if 1670 <= r["start"] <= 1685)
ranges = main_ranges[:ins + 1] + new_ranges + main_ranges[ins + 1:]

total = sum(x["end"] - x["start"] for x in ranges)
edl = {
    "version": 1,
    "sources": {SRC_KEY: SRC, NEW_KEY: NEW},
    "ranges": ranges,
    "grade": "none",
    "overlays": [],
    "total_duration_s": round(total, 2),
}
(base / "edl.json").write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"ranges: {len(ranges)} (main {len(main_ranges)} + new {len(new_ranges)})  total: {total/60:.1f} min ({total:.1f}s)")
print(f"new clip inserted after main range #{ins}: {main_ranges[ins]['quote'][:55]!r}")
bad = [(ranges[i]['end'], ranges[i+1]['start']) for i in range(len(ranges)-1)
       if ranges[i]['source'] == ranges[i+1]['source'] and ranges[i]['end'] > ranges[i+1]['start']]
print("overlaps(same src):", len(bad))
