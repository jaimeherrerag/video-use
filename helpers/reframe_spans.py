#!/usr/bin/env python3
"""reframe_spans.py - reframe solo en los tramos donde se ve cierta ventana.

Caso de uso (2026-10-07): recortar la barra de pestanas de Chrome SOLO cuando
Chrome esta en pantalla, y no en camara completa ni en la app de Claude.

  detect  Busca en el source cuando un "parche" de UI (p.ej. el boton "YouTube" de
          la barra de favoritos, que con Chrome maximizado siempre esta en el mismo
          lugar) se ve igual que en un frame de referencia. Pasa por los keyframes
          (rapido) y afina cada transicion a nivel de frame.
          -> <edit>/<nombre>.spans.json  (intervalos en tiempo de SOURCE)

  apply   Aplica un `reframe` a los ranges del EDL que caen dentro de esos tramos y
          parte los que cruzan un borde (la ventana cambia a mitad de frase).
          Va DESPUES de studio_cuts.py from-studio: los spans estan en tiempo de
          source, asi que sobreviven a cualquier edicion de cortes.

Uso:
    uv run python helpers/reframe_spans.py detect --source <video> --ref-time 1530 \\
        --patch 110,180,150,40 --out <edit>/chrome.spans.json
    uv run python helpers/reframe_spans.py apply --edl <edit>/edl.json \\
        --spans <edit>/chrome.spans.json --crop-top 80 [--out-size 1920x1080]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

UMBRAL = 12.0        # diferencia media (0-255, gris) para considerar el parche "igual"
PIEZA_MIN = 0.20     # al partir un range, piezas mas cortas se absorben en la vecina


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------
def frames_parche(src: Path, patch, ss: float | None = None, t: float | None = None,
                  solo_keyframes: bool = False):
    """(pts, parche gris) de cada frame. Con solo_keyframes decodifica solo I-frames."""
    x, y, w, h = patch
    cmd = ["ffmpeg", "-v", "info", "-nostats"]
    if solo_keyframes:
        cmd += ["-skip_frame", "nokey"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.3f}"]
    cmd += ["-i", str(src)]
    if t is not None:
        cmd += ["-t", f"{t:.3f}"]
    cmd += ["-map", "0:v:0", "-vf", f"crop={w}:{h}:{x}:{y},showinfo",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.run(cmd, capture_output=True)
    pts = [float(m) for m in re.findall(rb"pts_time:\s*([\d.]+)", p.stderr)]
    data = np.frombuffer(p.stdout, np.uint8)
    n = min(len(pts), len(data) // (w * h))
    frames = data[: n * w * h].reshape(n, h, w).astype(np.float32)
    base = ss or 0.0
    return [base + t_ for t_ in pts[:n]], frames


def cmd_detect(args) -> int:
    src = Path(args.source)
    patch = [int(v) for v in args.patch.split(",")]
    _, ref = frames_parche(src, patch, ss=args.ref_time, t=0.05)
    if not len(ref):
        sys.exit("no pude leer el frame de referencia")
    ref = ref[0]

    def igual(f) -> bool:
        return float(np.abs(f - ref).mean()) < args.umbral

    ts, fr = frames_parche(src, patch, solo_keyframes=True)
    clases = [igual(f) for f in fr]
    difs = sorted(round(float(np.abs(f - ref).mean())) for f in fr)
    print(f"{len(ts)} keyframes | diferencia con la referencia: "
          f"p10={difs[len(difs) // 10]} p50={difs[len(difs) // 2]} p90={difs[9 * len(difs) // 10]}"
          f" | umbral {args.umbral}")

    # Afinar cada cambio de clase entre dos keyframes, frame a frame
    bordes = []
    for i in range(1, len(ts)):
        if clases[i] != clases[i - 1]:
            sub_t, sub_f = frames_parche(src, patch, ss=ts[i - 1], t=ts[i] - ts[i - 1] + 0.01)
            corte = ts[i]
            for t_, f in zip(sub_t, sub_f):
                if igual(f) == clases[i]:
                    corte = t_
                    break
            bordes.append((corte, clases[i]))

    spans, ini = [], (0.0 if clases and clases[0] else None)
    for t_, entra in bordes:
        if entra and ini is None:
            ini = t_
        elif not entra and ini is not None:
            spans.append([round(ini, 3), round(t_, 3)])
            ini = None
    if ini is not None:
        spans.append([round(ini, 3), round(ts[-1] + 1.0, 3)])

    out = Path(args.out)
    out.write_text(json.dumps({"source": str(src).replace("\\", "/"), "ref_time": args.ref_time,
                               "patch": patch, "umbral": args.umbral, "spans": spans},
                              indent=2), encoding="utf-8")
    total = sum(b - a for a, b in spans)
    print(f"{len(spans)} tramos, {total / 60:.1f} min del source -> {out}")
    return 0


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def cmd_apply(args) -> int:
    edl_path = Path(args.edl)
    edl = json.loads(edl_path.read_text(encoding="utf-8-sig"))
    sp = json.loads(Path(args.spans).read_text(encoding="utf-8"))
    spans = sp["spans"]
    src_w, src_h = (int(v) for v in args.src_size.split("x"))
    out_w, out_h = (int(v) for v in args.out_size.split("x"))
    reframe = {"src_crop": {"x": 0, "y": args.crop_top, "w": src_w, "h": src_h - args.crop_top},
               "out_size": {"w": out_w, "h": out_h}, "fit": "crop-pad"}

    def dentro(t: float) -> bool:
        return any(a <= t < b for a, b in spans)

    nuevos, n_ref, n_split = [], 0, 0
    for r in edl["ranges"]:
        s, e = float(r["start"]), float(r["end"])
        cortes = sorted({s, e, *[x for a, b in spans for x in (a, b) if s < x < e]})
        piezas = [[a, b] for a, b in zip(cortes, cortes[1:])]
        # absorber piezas minimas en la vecina (no vale la pena un corte de 3 frames)
        k = 0
        while len(piezas) > 1 and k < len(piezas):
            if piezas[k][1] - piezas[k][0] < PIEZA_MIN:
                if k > 0:
                    piezas[k - 1][1] = piezas[k][1]
                else:
                    piezas[k + 1][0] = piezas[k][0]
                piezas.pop(k)
            else:
                k += 1
        n_split += len(piezas) - 1
        for a, b in piezas:
            q = dict(r)
            q["start"], q["end"] = round(a, 3), round(b, 3)
            q.pop("reframe", None)
            if dentro((a + b) / 2):
                q["reframe"] = reframe
                n_ref += 1
            nuevos.append(q)

    bk = edl_path.parent / "edl_backups"
    bk.mkdir(exist_ok=True)
    shutil.copy2(edl_path, bk / f"edl_antes_reframe_{datetime.now():%Y%m%d_%H%M%S}.json")
    edl["ranges"] = nuevos
    edl_path.write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{n_ref} de {len(nuevos)} ranges con reframe (crop-top {args.crop_top}) | "
          f"{n_split} ranges partidos en un borde de ventana -> {edl_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="reframe por tramos donde se ve una ventana")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--source", required=True)
    d.add_argument("--ref-time", type=float, required=True, help="s de un frame donde SE VE la ventana")
    d.add_argument("--patch", required=True, help="x,y,w,h en pixeles del source")
    d.add_argument("--umbral", type=float, default=UMBRAL)
    d.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--edl", required=True)
    a.add_argument("--spans", required=True)
    a.add_argument("--crop-top", type=int, required=True, help="px del source a quitar arriba")
    a.add_argument("--src-size", default="3840x2160")
    a.add_argument("--out-size", default="1920x1080")
    args = ap.parse_args()
    return cmd_detect(args) if args.cmd == "detect" else cmd_apply(args)


if __name__ == "__main__":
    sys.exit(main())
