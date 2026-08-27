"""Build edl.json from runs.json applying drop/trim decisions (keyed by run start
time, stable across re-indexing). Aggressive cut: keep only speech runs."""
import json
from pathlib import Path

base = Path(r"C:/dev/Video editing/projects/2026-06-01_video-36/edit")
SRC = r"C:/Users/Jaime Herrera/Videos/youtube/36 (01-06-26)/2026-06-01 21-51-41.mp4"
SRC_KEY = "2026-06-01 21-51-41"
HEAD, TAIL = 0.05, 0.08

runs = json.load(open(base / "runs.json", encoding="utf-8"))

# Runs to drop entirely (markers, failed attempts, repetitions, empties, phantom)
DROP = {
    49.06, 71.94,                      # repeticiones intro
    285.58, 290.52,                    # "instale--" + corta
    530.16, 533.83,                    # "sino que la verdadera razon..." + corta
    561.11, 582.87,                    # "a cambio..." 1er intento + corta
    647.59, 656.96,                    # "Y justo por esa razon..." + corta
    936.07, 942.49, 951.79,            # Opus4.8 1er intento + corto + "."
    1021.15, 1029.03, 1042.01, 1043.09,# sobreescribir 1er intento + enlace + "." + borra
    1141.78,                           # "Yo lo..."
    1198.54, 1208.30,                  # "otra cosa importante..." + corta
    1229.38, 1236.12,                  # "Y si probablemente..." + corta
    1493.46,                           # corta (Opus cuatro--)
    1611.10, 1619.56,                  # "no son los mejores...Grok cuatro--" + corta
    1782.77, 1798.86,                  # ". Corta eso" x2 en demo
    2074.67,                           # "Ok," suelto de 0.31s (blip)
    1911.38,                           # "me esta pidiendo per--"
    1969.98, 1971.82,                  # "Pero por cier--" + borra
    2218.14,                           # "."
    2514.70, 2519.55,                  # "Y un modelo gratuito..." + corta
    745.90,                            # "." cola CTA
}

# Trim overrides: start time -> {'start': X} and/or {'end': Y} (used as-is, no extra pad)
TRIM = {
    112.28: {"end": 115.00},    # "Llevamos meses usando Cloud Code" (drop "como el...")
    310.22: {"end": 313.10},    # "Tokens... suscripcion." (drop failed retake + stretch)
    413.96: {"end": 419.90},    # "...totalmente gratuita." (stretch trim)
    448.32: {"end": 454.55},    # "...los mas grandes." (stretch trim)
    684.74: {"end": 685.55},    # "...Opus 4.8." (stretch trim)
    1064.59: {"end": 1067.05},  # "...sus solicitudes." (drop "Vamos a cambiar, em, la llave... borra eso")
    1485.78: {"end": 1490.63},  # "...262 mil tokens." (drop "mientras que Opus cuatro--")
    1724.34: {"start": 1742.90},# drop phantom "Y aqui", keep "vemos que me acaba de dar una respuesta"
    1777.35: {"end": 1778.85},  # "...ultimos quince minutos," (drop "aqui me sale esta")
    1793.01: {"end": 1795.52},  # "Despues, vamos a probar otra cosa." (drop "Vamos a poder...")
    2574.38: {"end": 2576.18},  # "Y creeme que si lo usas bien," (drop "podria llegarte ahorrar--")
    2579.49: {"start": 2580.58},# drop "Corta eso.", keep "Podria llegar a ahorrarte hasta un 90%..."
}

ranges = []
for r in runs:
    st = r["s"]
    if st in DROP:
        continue
    txt = r["text"].strip(" .").strip()
    if not txt:
        continue
    start = r["s"] - HEAD
    end = r["e"] + TAIL
    if st in TRIM:
        ov = TRIM[st]
        if "start" in ov:
            start = ov["start"]
        if "end" in ov:
            end = ov["end"]
    ranges.append({"source": SRC_KEY, "start": round(start, 3), "end": round(end, 3),
                   "quote": r["text"][:90]})

# safety: ensure monotonic, no overlap
ranges.sort(key=lambda x: x["start"])
total = sum(x["end"] - x["start"] for x in ranges)
edl = {
    "version": 1,
    "sources": {SRC_KEY: SRC},
    "ranges": ranges,
    "grade": "warm_cinematic",
    "overlays": [],
    "total_duration_s": round(total, 2),
}
out = base / "edl.json"
import io
out.write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"ranges: {len(ranges)}  total: {total/60:.1f} min ({total:.1f}s)")
# overlap check
bad = [(ranges[i]['end'], ranges[i+1]['start']) for i in range(len(ranges)-1) if ranges[i]['end'] > ranges[i+1]['start']]
print("overlaps:", len(bad))
for b in bad[:5]: print("  ", b)
