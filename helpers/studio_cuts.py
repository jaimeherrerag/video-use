"""studio_cuts.py - revisión manual de cortes en HyperFrames Studio, ida y vuelta con el EDL.

Studio es el editor; render.py sigue siendo el renderizador. Cada range del EDL se vuelve
un <video> en <edit>/cortes/ sobre un proxy liviano de su fuente; Jaime recorta, parte,
borra o reordena en Studio, y from-studio reescribe el EDL con esos cambios.

    to-studio   --edit-dir E      EDL -> proyecto Studio (crea proxies la primera vez)
    from-studio --edit-dir E      Studio -> edl.json (respalda el anterior, imprime qué cambió)
    repack      --edit-dir E      from-studio + to-studio: cierra los huecos que deja un recorte

Studio no tiene ripple: recortar deja un hueco y alargar se encima con el clip siguiente.
Al exportar, los clips se concatenan en orden de inicio, así que huecos y encimados
desaparecen solos; `repack` solo sirve para volver a escuchar el corte continuo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from html import escape
from html.parser import HTMLParser
from pathlib import Path

# Raíz del repo "Video editing" (este archivo vive en video-use/helpers/)
REPO = Path(__file__).resolve().parents[2]
PRESET = REPO / "presets" / "glass-composition"   # de aquí salen package.json y hyperframes.json
STUDIO_DIR = "cortes"
LABEL_MAX = 70          # caracteres de la frase que se muestran como nombre del clip
EPS = 0.01              # cambios menores a esto no se reportan (redondeo de Studio)


# ── utilidades ────────────────────────────────────────────────────────────────

def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def fmt_t(s: float) -> str:
    m, sec = divmod(max(s, 0.0), 60)
    return f"{int(m):02d}:{sec:05.2f}"


def range_words(transcripts: Path, source: str, start: float, end: float, cache: dict) -> str:
    """Texto del transcript dentro del range, para nombrar el clip en el timeline."""
    if source not in cache:
        t = transcripts / f"{source}.json"
        cache[source] = load_json(t).get("words", []) if t.exists() else []
    words = [(w.get("text") or "").strip() for w in cache[source]
             if w.get("type", "word") == "word" and w.get("start") is not None
             and w["start"] >= start - 0.05 and w.get("end", w["start"]) <= end + 0.05]
    return " ".join(w for w in words if w)


def make_proxy(src: Path, dst: Path) -> None:
    """Proxy 540p (lado corto) con keyframes densos para que Studio busque rápido.
    Sin -map: ffmpeg elige la misma pista de audio que render.py (selección por defecto)."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    print(f"  proxy: {src.name} -> {dst.name} (~2 s por minuto de fuente)…", flush=True)
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src),
           "-vf", "scale='if(gt(iw,ih),-2,540)':'if(gt(iw,ih),540,-2)'",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
           "-g", "15", "-keyint_min", "15", "-sc_threshold", "0",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)


def probe_wh(p: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


# ── to-studio ─────────────────────────────────────────────────────────────────

def to_studio(edit: Path, edl_path: Path, force: bool) -> None:
    edl = load_json(edl_path)
    out = edit / STUDIO_DIR
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # Protección: no pisar ediciones de Studio que todavía no se exportaron
    index = out / "index.html"
    stamp = out / ".generated.sha1"
    if index.exists() and stamp.exists() and not force:
        if sha(index.read_text(encoding="utf-8")) != stamp.read_text().strip():
            sys.exit("Hay cambios de Studio sin exportar en cortes/index.html.\n"
                     "Corre `from-studio` (o `repack`) primero, o usa --force para descartarlos.")

    # Proxies: uno por fuente, se reutilizan entre corridas
    proxies = {}
    for key, src in edl["sources"].items():
        src_path = Path(src)
        if not src_path.is_absolute():
            src_path = (edit / src_path).resolve()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        dst = assets / f"proxy_{safe}.mp4"
        make_proxy(src_path, dst)
        proxies[key] = dst

    # Lienzo según la orientación de la primera fuente (16:9 o 9:16)
    pw, ph = probe_wh(next(iter(proxies.values())))
    W, H = (1920, 1080) if pw >= ph else (1080, 1920)

    # Tamaño real de cada fuente: el `reframe` del EDL viene en píxeles del SOURCE
    src_wh = {}
    for key, src in edl["sources"].items():
        p = Path(src) if Path(src).is_absolute() else (edit / src).resolve()
        src_wh[key] = probe_wh(p)

    def reframe_css(r: dict) -> str:
        """Simula en Studio el `reframe` de render.py (el proxy no lo lleva).

        Solo visual: recorta con clip-path y mueve/escala con transform para que el
        clip se vea con el encuadre del render. from-studio no lee el style.
        """
        rf = r.get("reframe")
        if not rf or rf.get("fit") not in ("crop-pad", "crop-scale"):
            return ""
        sw, sh = src_wh[r["source"]]
        k = W / sw                                   # px de lienzo por px de source
        c = rf["src_crop"]
        cx, cy, cw, ch = c["x"] * k, c["y"] * k, c["w"] * k, c["h"] * k
        ow, oh = rf.get("out_size", {}).get("w", W), rf.get("out_size", {}).get("h", H)
        if rf["fit"] == "crop-pad":                  # reduce si no cabe, centra, nunca amplía
            s = min(1.0, ow / c["w"], oh / c["h"])
            tw, th = c["w"] * s * W / ow, c["h"] * s * H / oh
            mx = my = tw / cw
        else:                                        # crop-scale: llena la salida
            tw, th = W, H
            mx, my = tw / cw, th / ch
        tx, ty = (W - tw) / 2, (H - th) / 2
        inset = f"{cy:.1f}px {W - cx - cw:.1f}px {H - cy - ch:.1f}px {cx:.1f}px"
        return (f' style="clip-path: inset({inset}); transform-origin: 0 0;'
                f' transform: translate({tx - cx * mx:.1f}px, {ty - cy * my:.1f}px)'
                f' scale({mx:.4f}, {my:.4f});"')

    # Un <video> por range, con inicio absoluto (sin referencias: Studio las pierde al editar)
    cache: dict = {}
    clips, t = [], 0.0
    for i, r in enumerate(edl["ranges"]):
        dur = round(r["end"] - r["start"], 3)   # redondeo: evita encimados de 1e-15 s
        text = range_words(edit / "transcripts", r["source"], r["start"], r["end"], cache) \
            or r.get("beat") or r.get("note") or ""
        label = f"{i + 1} · {text}"
        if len(label) > LABEL_MAX:
            label = label[:LABEL_MAX - 1].rstrip() + "…"
        clips.append(
            f'    <video id="r{i + 1:03d}" src="assets/{proxies[r["source"]].name}"'
            f' data-range="{i}" data-source="{escape(r["source"])}"'
            f' data-timeline-label="{escape(label)}"\n'
            f'      data-start="{t:.3f}" data-duration="{dur:.3f}" data-media-start="{r["start"]:.3f}"'
            f' data-has-audio="true" data-track-index="0"{reframe_css(r)} playsinline></video>')
        t = round(t + dur, 3)

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width={W}, height={H}"/>
  <title>Revision de cortes</title>
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
  <style>
    html, body {{ margin: 0; width: {W}px; height: {H}px; overflow: hidden; background: #000; }}
    #root {{ position: relative; width: {W}px; height: {H}px; overflow: hidden; }}
    #root video {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }}
  </style>
</head>
<body>
  <!-- Generado por video-use/helpers/studio_cuts.py — editar los cortes en Studio, no aquí.
       No pasa `hyperframes check` (timeline GSAP vacío) y no hace falta: nunca se renderiza con HF. -->
  <div id="root" data-composition-id="cortes" data-start="0" data-width="{W}" data-height="{H}">
{chr(10).join(clips)}
  </div>
  <script>
    window.__timelines = window.__timelines || {{}};
    window.__timelines["cortes"] = gsap.timeline({{ paused: true }});
  </script>
</body>
</html>
"""
    index.write_text(html, encoding="utf-8")
    stamp.write_text(sha(html))
    # El EDL base guarda los campos de cada range (grade, reframe, censor…) para la vuelta
    (out / "edl_base.json").write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")

    # package.json / hyperframes.json del preset: misma versión pineada de HyperFrames
    pkg = load_json(PRESET / "package.json")
    pkg["name"] = "cortes"
    (out / "package.json").write_text(json.dumps(pkg, indent=2), encoding="utf-8")
    shutil.copy(PRESET / "hyperframes.json", out / "hyperframes.json")

    print(f"{len(clips)} cortes, {fmt_t(t)} en total -> {index}")
    if edl.get("overlays"):
        print("  OJO: el EDL trae overlays con start_in_output; si cambias cortes antes de ellos,"
              " quedan desfasados (las animaciones van DESPUÉS de revisar cortes).")


# ── from-studio ───────────────────────────────────────────────────────────────

class VideoClips(HTMLParser):
    """Junta los atributos de cada <video> con data-range, en orden de documento."""
    def __init__(self) -> None:
        super().__init__()
        self.clips: list[dict] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "video" and "data-range" in a:
            a["_order"] = len(self.clips)
            self.clips.append(a)


def from_studio(edit: Path, out_path: Path, dry_run: bool) -> dict:
    sdir = edit / STUDIO_DIR
    base = load_json(sdir / "edl_base.json")
    parser = VideoClips()
    parser.feed((sdir / "index.html").read_text(encoding="utf-8"))

    clips = []
    for c in parser.clips:
        if "data-hidden" in c and c.get("data-hidden") not in ("false", "0"):
            continue    # clip ocultado en Studio = descartado
        start = float(c.get("data-start", 0))
        dur = float(c["data-duration"])
        ms = float(c.get("data-media-start", 0))
        clips.append({"idx": int(c["data-range"]), "src": c.get("data-source"),
                      "t": start, "in": ms, "out": ms + dur, "order": c["_order"]})
    clips.sort(key=lambda c: (c["t"], c["order"]))

    # Rangos nuevos = campos del range original + entrada/salida que dejó Studio
    new_ranges, pieces, notes_by = [], {}, {}
    prev_end, prev_idx = None, -1
    for c in clips:
        orig = base["ranges"][c["idx"]]
        r = dict(orig)
        r["source"] = c["src"] or orig["source"]
        r["start"], r["end"] = round(c["in"], 3), round(c["out"], 3)
        new_ranges.append(r)
        pieces.setdefault(c["idx"], []).append(r)
        notes = notes_by.setdefault(c["idx"], [])
        if c["idx"] < prev_idx:
            notes.append("movido de lugar")
        if prev_end is not None and c["t"] < prev_end - 0.05:
            notes.append("se encimaba con el anterior: se concatena en orden")
        prev_end, prev_idx = c["t"] + (c["out"] - c["in"]), max(prev_idx, c["idx"])

    # Por range original: entrada del primer pedazo, salida del último y tramos quitados en medio
    for i, ps in pieces.items():
        orig, ps = base["ranges"][i], sorted(ps, key=lambda r: r["start"])
        d_in, d_out = ps[0]["start"] - orig["start"], ps[-1]["end"] - orig["end"]
        extra = []
        if abs(d_in) > EPS:
            extra.append(f"entrada {d_in:+.2f}s ({'recorta' if d_in > 0 else 'recupera'})")
        if abs(d_out) > EPS:
            extra.append(f"salida {d_out:+.2f}s ({'recupera' if d_out > 0 else 'recorta'})")
        if len(ps) > 1:
            extra.append(f"partido en {len(ps)}")
            for a, b in zip(ps, ps[1:]):
                gap = b["start"] - a["end"]
                if gap > EPS:
                    extra.append(f"quita {gap:.2f}s de en medio (fuente {a['end']:.2f}–{b['start']:.2f})")
                elif gap < -EPS:
                    extra.append(f"repite {-gap:.2f}s (fuente {b['start']:.2f}–{a['end']:.2f})")
        notes_by[i] = extra + notes_by[i]

    # Reporte legible: solo lo que cambió, en el orden del video nuevo
    cache: dict = {}
    def label(i: int) -> str:
        o = base["ranges"][i]
        txt = range_words(edit / "transcripts", o["source"], o["start"], o["end"], cache) or o.get("beat", "")
        return f"{i + 1} · {txt[:55]}"

    lines = [f"  {label(i)}\n      " + "; ".join(dict.fromkeys(n))
             for i, n in notes_by.items() if n]
    deleted = [i for i in range(len(base["ranges"])) if i not in pieces]
    for i in deleted:
        lines.append(f"  {label(i)}\n      ELIMINADO")

    old_total = sum(r["end"] - r["start"] for r in base["ranges"])
    new_total = sum(r["end"] - r["start"] for r in new_ranges)
    print(f"Cortes: {len(base['ranges'])} -> {len(new_ranges)}   "
          f"duración: {fmt_t(old_total)} -> {fmt_t(new_total)} ({new_total - old_total:+.2f}s)")
    print("\n".join(lines) if lines else "  Sin cambios respecto al EDL base.")

    new_edl = dict(base)
    new_edl["ranges"] = new_ranges
    if not dry_run:
        if out_path.exists():
            bak = edit / "edl_backups" / f"edl_{datetime.now():%Y%m%d-%H%M%S}.json"
            bak.parent.mkdir(exist_ok=True)
            shutil.copy(out_path, bak)
            print(f"  respaldo del EDL anterior: {bak.relative_to(edit)}")
        out_path.write_text(json.dumps(new_edl, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  escrito: {out_path}")
    return new_edl


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["to-studio", "from-studio", "repack"])
    ap.add_argument("--edit-dir", required=True, type=Path)
    ap.add_argument("--edl", type=Path, help="default: <edit-dir>/edl.json")
    ap.add_argument("--force", action="store_true", help="to-studio: descartar cambios de Studio sin exportar")
    ap.add_argument("--dry-run", action="store_true", help="from-studio: solo reportar, no escribir")
    args = ap.parse_args()

    edit = args.edit_dir.resolve()
    edl_path = (args.edl or edit / "edl.json").resolve()
    if args.cmd == "to-studio":
        to_studio(edit, edl_path, args.force)
    elif args.cmd == "from-studio":
        from_studio(edit, edl_path, args.dry_run)
    else:  # repack
        from_studio(edit, edl_path, dry_run=False)
        to_studio(edit, edl_path, force=True)


if __name__ == "__main__":
    main()
