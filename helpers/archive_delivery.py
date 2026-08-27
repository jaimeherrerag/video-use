"""Archiva un video terminado siguiendo la convencion del canal.

Hace tres cosas que Jaime venia haciendo a mano despues de cada entrega:

  1. Renombra el entregable a un slug con palabras clave (para el algoritmo).
  2. Crea `Videos/youtube/<N> (DD-MM-YY)/` con el siguiente numero de video.
  3. Mueve ahi el entregable Y los archivos crudos (.mkv + .mp4 de OBS).

Ademas reescribe las rutas de `sources` en el EDL para que el proyecto siga
siendo re-renderizable despues de mover el footage (si no, render.py falla con
"No such file or directory" y hay que ir a buscar el .mp4 a mano).

Uso:
    uv run python helpers/archive_delivery.py \
        --edit-dir "<proyecto>/edit" \
        --slug deepseek_harness_claude_code_gratis \
        [--final final_1080p30_speed108.mp4] [--date 22-08-26] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

VIDEOS = Path.home() / "Videos"
YT = VIDEOS / "youtube"
# carpetas tipo "43 (15-08-26)"; el año va en 2 digitos (el 4-digitos de la 37 es un desliz)
FOLDER_RE = re.compile(r"^(\d+)\s*\((\d{2}-\d{2}-\d{2,4})\)$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def next_number() -> int:
    if not YT.is_dir():
        sys.exit(f"no existe {YT}")
    nums = [int(m.group(1)) for d in YT.iterdir() if d.is_dir()
            for m in [FOLDER_RE.match(d.name)] if m]
    if not nums:
        sys.exit(f"no encontre carpetas con formato '<N> (DD-MM-YY)' en {YT}")
    return max(nums) + 1


def raw_siblings(src: Path) -> list[Path]:
    """El .mp4 del EDL y su .mkv hermano (OBS graba mkv y luego remuxea)."""
    out = [src] if src.exists() else []
    for ext in (".mkv", ".mov"):
        sib = src.with_suffix(ext)
        if sib.exists():
            out.append(sib)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Archiva un entregable en Videos/youtube/")
    ap.add_argument("--edit-dir", type=Path, required=True)
    ap.add_argument("--slug", type=str, required=True,
                    help="nombre con palabras clave, snake_case (ej. deepseek_harness_gratis)")
    ap.add_argument("--final", type=str, default=None,
                    help="nombre del entregable dentro de edit/ (default: el .mp4 que empiece por 'final')")
    ap.add_argument("--date", type=str, default=None,
                    help="DD-MM-YY (default: la fecha de modificacion del entregable, "
                         "que es cuando realmente se termino — no 'hoy')")
    ap.add_argument("--number", type=int, default=None, help="fuerza el numero de video")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SLUG_RE.match(args.slug):
        sys.exit(f"slug invalido: {args.slug!r} (usa minusculas y guiones bajos)")

    edit = args.edit_dir.resolve()
    if not edit.is_dir():
        sys.exit(f"no existe {edit}")

    # --- entregable ---
    if args.final:
        final = edit / args.final
    else:
        cands = sorted(edit.glob("final*.mp4"))
        if len(cands) != 1:
            sys.exit(f"esperaba 1 'final*.mp4' en {edit}, encontre {len(cands)}: "
                     f"{[c.name for c in cands]} — usa --final")
        final = cands[0]
    if not final.exists():
        sys.exit(f"no existe el entregable: {final}")

    # --- fuentes del EDL ---
    edl_path = edit / "edl.json"
    edl = json.loads(edl_path.read_text(encoding="utf-8")) if edl_path.exists() else None
    raws: list[Path] = []
    if edl:
        for p in edl.get("sources", {}).values():
            src = Path(p)
            # solo se archivan las fuentes que viven en Videos/ (los derivados
            # tipo source_mixed.mov se quedan en el proyecto)
            try:
                src.relative_to(VIDEOS)
            except ValueError:
                continue
            for f in raw_siblings(src):
                if f not in raws:
                    raws.append(f)

    n = args.number or next_number()
    # la fecha de finalizacion es la del render, no la del dia en que se archiva
    d = args.date or datetime.fromtimestamp(final.stat().st_mtime).strftime("%d-%m-%y")
    dest = YT / f"{n} ({d})"
    new_final = dest / f"{args.slug}{final.suffix}"

    print(f"destino: {dest}")
    print(f"  entregable: {final.name}  ->  {new_final.name}  "
          f"({final.stat().st_size/1e6:.0f} MB)")
    for f in raws:
        print(f"  crudo     : {f.name}  ({f.stat().st_size/1e9:.2f} GB)")
    if not raws:
        print("  (sin crudos que mover — revisa las sources del EDL)")

    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"\n{dest} ya existe y no esta vacia — aborto")
    if args.dry_run:
        print("\n--dry-run: no se movio nada")
        return

    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(final), str(new_final))
    print(f"\nmovido: {new_final}")
    moved = {}
    for f in raws:
        tgt = dest / f.name
        shutil.move(str(f), str(tgt))
        moved[str(f)] = str(tgt)
        print(f"movido: {tgt}")

    # --- reapuntar el EDL a las rutas nuevas (JSON SIN BOM: json.loads revienta con el) ---
    if edl and moved:
        changed = 0
        for k, v in list(edl.get("sources", {}).items()):
            tgt = moved.get(str(Path(v)))
            if tgt:
                edl["sources"][k] = str(Path(tgt)).replace("\\", "/")
                changed += 1
        if changed:
            edl_path.write_text(json.dumps(edl, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            print(f"\nedl.json: {changed} source(s) reapuntadas a la carpeta nueva")
        for sub in sorted(edit.glob("stitch/*_edl.json")):
            s = json.loads(sub.read_text(encoding="utf-8"))
            ch = 0
            for k, v in list(s.get("sources", {}).items()):
                tgt = moved.get(str(Path(v)))
                if tgt:
                    s["sources"][k] = str(Path(tgt)).replace("\\", "/")
                    ch += 1
            if ch:
                sub.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{sub.name}: {ch} source(s) reapuntadas")

    print(f"\nOK — falta subir la miniatura como {dest / (args.slug + '.png')}")


if __name__ == "__main__":
    main()
