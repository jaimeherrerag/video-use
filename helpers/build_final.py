#!/usr/bin/env python3
"""build_final.py - del Studio maestro al entregable, sin pasos a mano.

Flujo (regla de oro): Jaime edita en <edit>/studio/ -> `studio_maestro.py from-studio` deja
edl.json, stitch.json y los slots al dia -> este script arma el video:

  1. Parte el EDL en tramos: los SLOTS de HyperFrames (edl.studio.slots, p.ej. la intro con
     tarjetas glass) y lo normal entre ellos. Cada tramo -> <edit>/stitch/<nombre>/edl.json,
     con sus overlays (CTA) en tiempo relativo al tramo.
  2. Overlays con `comp` (CTA): re-renderiza su green screen si la composicion cambio.
  3. render.py --pcm de cada tramo (se salta si su EDL no cambio desde el ultimo build).
  4. Slots: base del slot (-g 6), duraciones del index.html, karaoke de frases clave
     (frases.txt + build_karaoke --frases) y render de HyperFrames (se salta si nada cambio).
  5. stitch.json: partes en orden -> stitch_final.py -> preview_final.mp4.

    uv run python helpers/build_final.py --edit-dir <edit> [--forzar]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HELP = Path(__file__).parent
REPO = HELP.parents[1]
PY = [sys.executable]


def sha_de(*paths: Path) -> str:
    h = hashlib.sha1()
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix in (".html", ".json", ".txt", ".png", ".jpg", ".webp", ".svg"):
                    h.update(f.name.encode())
                    h.update(f.read_bytes())
        elif p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def cambio(marca: Path, firma: str, forzar: bool) -> bool:
    if forzar or not marca.exists() or marca.read_text().strip() != firma:
        return True
    return False


def run(cmd: list, cwd: Path | None = None) -> None:
    print("  $ " + " ".join(str(c) for c in cmd[:6]) + (" ..." if len(cmd) > 6 else ""), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, shell=False)
    if r.returncode != 0:
        sys.exit(f"fallo: {' '.join(str(c) for c in cmd)}")


def dur_video(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=duration", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout
    return float(out.strip().splitlines()[0])


def ventana_slot(edit: Path, s: dict) -> tuple[float, float]:
    """[inicio, fin] del slot en la salida: su t y el fin del host que termina mas tarde."""
    idx = (edit / s["dir"] / "index.html").read_text(encoding="utf-8")
    fin = 0.0
    for m in re.finditer(r"<div[^>]*data-composition-src[^>]*>", idx):
        tag = m.group(0)
        st = float(re.search(r'data-start="([\d.]+)"', tag).group(1))
        du = float(re.search(r'data-duration="([\d.]+)"', tag).group(1))
        fin = max(fin, st + du)
    return s["t"], s["t"] + fin


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edit-dir", required=True, type=Path)
    ap.add_argument("--forzar", action="store_true", help="re-renderizar todo aunque no haya cambiado")
    args = ap.parse_args()
    edit = args.edit_dir.resolve()
    edl = json.loads((edit / "edl.json").read_text(encoding="utf-8"))
    stitch_dir = edit / "stitch"
    spec_path = stitch_dir / "stitch.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    fps = int(spec.get("fps", 30))

    # --- 1. tramos ------------------------------------------------------------------
    slots = sorted((edl.get("studio") or {}).get("slots", []), key=lambda s: s["t"])
    vents = [(ventana_slot(edit, s), s) for s in slots]
    tramos, cur, t = [], None, 0.0
    for r in edl["ranges"]:
        slot = next((s for (a, b), s in vents if a - 0.05 <= t < b - 0.05), None)
        clave = slot["dir"] if slot else None
        if cur is None or cur["clave"] != clave:
            cur = {"clave": clave, "slot": slot, "ranges": [], "t0": t}
            tramos.append(cur)
        cur["ranges"].append(r)
        t += r["end"] - r["start"]
    n_plain = 0
    for tr in tramos:
        if tr["slot"]:
            tr["nombre"] = Path(tr["slot"]["dir"]).name.replace("slot_", "") or "slot"
        else:
            n_plain += 1
            tr["nombre"] = "rest" if n_plain == 1 else f"rest{n_plain}"
        tr["dur"] = sum(r["end"] - r["start"] for r in tr["ranges"])
    print("tramos: " + " | ".join(f"{tr['nombre']} {tr['t0']:.1f}-{tr['t0'] + tr['dur']:.1f}s"
                                   f" ({len(tr['ranges'])} cortes)" for tr in tramos))

    # --- 2. overlays con composicion (CTA): green screen al dia -------------------------
    for ov in edl.get("overlays") or []:
        if not ov.get("comp"):
            continue
        cdir = (edit / ov["comp"]).parent.parent       # animations/cta/
        marca = cdir / ".render.sha1"
        firma = sha_de(cdir / "compositions", cdir / "index.html")
        if cambio(marca, firma, args.forzar) or not (edit / ov["file"]).exists():
            print(f"overlay {cdir.name}: render green screen")
            run(["npm", "run", "render", "--", "--fps", str(fps), "--quality", "high", "-o",
                 Path(ov["file"]).name], cwd=cdir) if sys.platform != "win32" else \
                run(["cmd", "/c", "npm", "run", "render", "--", "--fps", str(fps), "--quality", "high",
                     "-o", Path(ov["file"]).name], cwd=cdir)
            marca.write_text(firma)

    # --- 3. render.py --pcm de cada tramo -----------------------------------------------
    parts = []
    for tr in tramos:
        d = stitch_dir / tr["nombre"]
        d.mkdir(parents=True, exist_ok=True)
        sub = {k: v for k, v in edl.items() if k not in ("ranges", "overlays", "studio")}
        sub["ranges"] = tr["ranges"]
        sub["overlays"] = []
        for ov in edl.get("overlays") or []:
            if tr["t0"] <= ov["start_in_output"] < tr["t0"] + tr["dur"]:
                o = dict(ov)
                o["file"] = str((edit / ov["file"]).resolve()).replace("\\", "/")
                o["start_in_output"] = round(ov["start_in_output"] - tr["t0"], 3)
                o.pop("comp", None)
                sub["overlays"].append(o)
        if tr["slot"] and sub["overlays"]:
            print(f"  aviso: overlay dentro del slot {tr['nombre']}: render.py lo compone igual")
        txt = json.dumps(sub, ensure_ascii=False, indent=2)
        (d / "edl.json").write_text(txt, encoding="utf-8")
        pcm = d / f"{tr['nombre']}_pcm.mp4"
        firma = hashlib.sha1((txt + sha_de(*[(edit / o["file"]) for o in edl.get("overlays") or []])).encode()).hexdigest()
        if cambio(d / ".edl.sha1", firma, args.forzar) or not pcm.exists():
            print(f"tramo {tr['nombre']}: render.py --pcm")
            run(PY + [HELP / "render.py", d / "edl.json", "-o", pcm, "--fps", fps, "--high-quality",
                      "--pcm", "--no-subtitles"])
            (d / ".edl.sha1").write_text(firma)
        else:
            print(f"tramo {tr['nombre']}: sin cambios, se reutiliza {pcm.name}")
        tr["pcm"] = pcm

    # --- 4. slots de HyperFrames ----------------------------------------------------------
    for tr in tramos:
        if not tr["slot"]:
            parts.append({"video": str(tr["pcm"].relative_to(stitch_dir)).replace("\\", "/"),
                          "audio": str(tr["pcm"].relative_to(stitch_dir)).replace("\\", "/")})
            continue
        sdir = (edit / tr["slot"]["dir"]).resolve()
        base = sdir / "assets" / "base.mp4"
        if cambio(sdir / ".base.sha1", (tr["pcm"].parent / ".edl.sha1").read_text(), args.forzar) or not base.exists():
            run(["ffmpeg", "-v", "error", "-y", "-i", tr["pcm"], "-c:v", "libx264", "-preset", "slow",
                 "-crf", "16", "-pix_fmt", "yuv420p", "-g", "6", "-keyint_min", "6", "-sc_threshold", "0",
                 "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", base])
            (sdir / ".base.sha1").write_text((tr["pcm"].parent / ".edl.sha1").read_text())
            frases = sdir / "frases.txt"
            if frases.exists():
                print(f"slot {tr['nombre']}: karaoke de frases clave")
                key = next(iter(edl["sources"]))
                run(PY + [HELP / "build_karaoke.py", "--transcript", edit / "transcripts" / f"{key}.json",
                          "--edl", tr["pcm"].parent / "edl.json",
                          "--template", REPO / "presets" / "glass-composition" / "compositions" / "subtitles.html",
                          "--output", sdir / "compositions" / "subtitles.html", "--frases", frases,
                          "--max-per-pill", "9"])
        # duraciones del slot = las del video base (root, video, audio, karaoke, padding)
        D = dur_video(base)
        idx = sdir / "index.html"
        txt = idx.read_text(encoding="utf-8")
        txt = re.sub(r'(data-composition-id="[^"]*" data-start="0" data-duration=")[\d.]+(")', rf"\g<1>{D:.3f}\g<2>", txt, count=1)
        txt = re.sub(r'(id="face-video"[^>]*data-duration=")[\d.]+(")', rf"\g<1>{D:.3f}\g<2>", txt)
        txt = re.sub(r'(id="base-audio"[^>]*data-duration=")[\d.]+(")', rf"\g<1>{D:.3f}\g<2>", txt)
        txt = re.sub(r'(id="subs-layer"[^>]*data-duration=")[\d.]+(")', rf"\g<1>{D:.3f}\g<2>", txt)
        txt = re.sub(r'(tl\.to\(\{\},\s*\{\s*duration:\s*)[\d.]+', rf"\g<1>{D:.3f}", txt)
        idx.write_text(txt, encoding="utf-8")
        render = sdir / f"render_{tr['nombre']}.mp4"
        firma = sha_de(sdir / "compositions", idx, sdir / ".base.sha1")
        if cambio(sdir / ".render.sha1", firma, args.forzar) or not render.exists():
            print(f"slot {tr['nombre']}: render de HyperFrames ({D:.1f}s)")
            npm = ["cmd", "/c", "npm"] if sys.platform == "win32" else ["npm"]
            run(npm + ["run", "render", "--", "--fps", str(fps), "--quality", "high", "-o", render.name], cwd=sdir)
            (sdir / ".render.sha1").write_text(firma)
        else:
            print(f"slot {tr['nombre']}: sin cambios, se reutiliza {render.name}")
        parts.append({"video": str(Path("..") / render.relative_to(edit)).replace("\\", "/"),
                      "audio": str(tr["pcm"].relative_to(stitch_dir)).replace("\\", "/")})

    # --- 5. stitch ------------------------------------------------------------------------
    spec["parts"] = parts
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    run(PY + [HELP / "stitch_final.py", spec_path])
    return 0


if __name__ == "__main__":
    sys.exit(main())
