#!/usr/bin/env python3
"""studio_maestro.py - UN proyecto de HyperFrames Studio por video con TODOS los edits.

REGLA DE ORO (CLAUDE.md, 2026-10-08): todo edit de la IA debe quedar visible y editable en
Studio antes del render final. Este proyecto (<edit>/studio/) es la mesa de trabajo unica:

  clips       un <video> por range del EDL sobre el proxy 540p (cortar, partir, borrar,
              reordenar, como studio_cuts)
  encuadre    el reframe de cada range (recorte de pestanas, zooms) como tamano/translate/
              clip-path del clip: se ajusta con las manijas del canvas (geometria_studio.py)
  blurs       cada `censor` del EDL como un rectangulo editable (mover/redimensionar)
  slots       las sub-composiciones de los slots HF (tarjetas, karaoke de la intro...) en su
              tiempo, editables por dentro y en el timeline
  overlays    el CTA y demas overlays con `comp` como sub-composicion (mismo HTML que se
              renderiza en green screen)
  audio       musica y SFX del stitch.json con su volumen relativo a la voz (mezcla_studio)

    to-studio   --edit-dir E [--force]   EDL + stitch.json + slots -> E/studio/
    from-studio --edit-dir E [--dry-run] Studio -> edl.json, stitch.json, slots (con respaldos)

Studio reproduce a 1x (el entregable va a `speed`). Recortar un clip en Studio deja hueco
(no hace ripple): todo tiempo de Studio se convierte al timeline de SALIDA por el clip que
tiene debajo, asi el CTA, las tarjetas y los SFX siguen al contenido.
"""

from __future__ import annotations

import argparse
import re
import json
import shutil
import sys
from datetime import datetime
from html import escape
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from geometria_studio import (W, H, css_de_place, fuente_a_lienzo, lienzo_a_fuente,  # noqa: E402
                              place_de_css, place_de_reframe, rect_div, reframe_de_place)
from mezcla_studio import clips_audio, leer_audio  # noqa: E402
from stitch_final import cargar_partes, medir_ganancia  # noqa: E402
from studio_cuts import fmt_t, load_json, make_proxy, probe_wh, range_words, sha  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PRESET = REPO / "presets" / "glass-composition"
DIR = "studio"
LABEL_MAX = 70


class Elementos(HTMLParser):
    """Atributos de todos los elementos con id, en orden de documento."""
    def __init__(self) -> None:
        super().__init__()
        self.el: list[tuple[str, dict]] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            a["_order"] = len(self.el)
            self.el.append((tag, a))


def oculto(a: dict) -> bool:
    return "data-hidden" in a and a.get("data-hidden") not in ("false", "0")


def hosts_de_slot(index: Path) -> list[dict]:
    p = Elementos()
    p.feed(index.read_text(encoding="utf-8"))
    return [a for tag, a in p.el if a.get("data-composition-src")]


# ── to-studio ─────────────────────────────────────────────────────────────────

def to_studio(edit: Path, force: bool) -> None:
    edl = load_json(edit / "edl.json")
    spec_path = edit / "stitch" / "stitch.json"
    spec = load_json(spec_path) if spec_path.exists() else None
    out = edit / DIR
    assets, comps = out / "assets", out / "compositions"
    assets.mkdir(parents=True, exist_ok=True)
    comps.mkdir(parents=True, exist_ok=True)

    index, stamp = out / "index.html", out / ".generated.sha1"
    if index.exists() and stamp.exists() and not force:
        if sha(index.read_text(encoding="utf-8")) != stamp.read_text().strip():
            sys.exit("Hay cambios de Studio sin exportar en studio/index.html.\n"
                     "Corre `from-studio` primero, o usa --force para descartarlos.")

    # proxies (se reutiliza el de la revision de cortes si existe) y tamano real de cada fuente
    proxies, dims = {}, {}
    for key, src in edl["sources"].items():
        sp = Path(src) if Path(src).is_absolute() else (edit / src).resolve()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        dst = assets / f"proxy_{safe}.mp4"
        prev = edit / "cortes" / "assets" / dst.name
        if not dst.exists() and prev.exists():
            shutil.copy2(prev, dst)
        make_proxy(sp, dst)
        proxies[key], dims[key] = dst.name, probe_wh(sp)

    lines, t, cache = [], 0.0, {}
    clip_win = []                                   # (t0, t1, place) por range, en Studio
    for i, r in enumerate(edl["ranges"]):
        d = round(r["end"] - r["start"], 3)
        sw, sh = dims[r["source"]]
        place = place_de_reframe(r.get("reframe"), sw, sh)
        css = css_de_place(place, sw, sh)
        text = range_words(edit / "transcripts", r["source"], r["start"], r["end"], cache) or r.get("beat", "")
        label = f"{i + 1} · {text}"
        label = label if len(label) <= LABEL_MAX else label[:LABEL_MAX - 1].rstrip() + "…"
        style = f' style="{css}"' if css else ""
        lines.append(
            f'    <video id="c{i + 1:03d}" src="assets/{proxies[r["source"]]}" data-range="{i}"'
            f' data-source="{escape(r["source"])}" data-timeline-label="{escape(label)}"\n'
            f'      data-start="{t:.3f}" data-duration="{d:.3f}" data-media-start="{r["start"]:.3f}"'
            f' data-has-audio="true" data-track-index="0"{style} playsinline></video>')
        clip_win.append((t, t + d, place))
        t = round(t + d, 3)
    total = t

    # blurs: un rectangulo por censor y por range, en coordenadas del lienzo
    nb = 0
    for i, r in enumerate(edl["ranges"]):
        for c in r.get("censor") or []:
            t0, t1, place = clip_win[i]
            sw, sh = dims[r["source"]]
            x, y, wd, ht = fuente_a_lienzo(place, (c["x"], c["y"], c["w"], c["h"]), sw, sh)
            nb += 1
            lines.append(
                f'    <div id="blur-{nb:02d}" class="clip blur-censura" data-timeline-label="Blur {nb} (censura)"\n'
                f'      data-start="{t0:.3f}" data-duration="{t1 - t0:.3f}" data-track-index="1"\n'
                f'      style="position: absolute; left: {x:.1f}px; top: {y:.1f}px; width: {wd:.1f}px; '
                f'height: {ht:.1f}px;"></div>')

    # slots HF (tarjetas, karaoke...): sus hosts en el timeline maestro
    mapa_slots = {}
    for s in (edl.get("studio") or {}).get("slots", []):
        sdir = (edit / s["dir"]).resolve()
        for f in (sdir / "compositions").glob("*.html"):
            shutil.copy2(f, comps / f.name)
        for f in (sdir / "assets").iterdir():
            if f.is_file() and f.name not in ("base.mp4", "README.md"):
                shutil.copy2(f, assets / f.name)
        for h in hosts_de_slot(sdir / "index.html"):
            st, du = float(h["data-start"]) + s["t"], float(h["data-duration"])
            du = min(du, total - st)
            lines.append(
                f'    <div id="{h["id"]}" class="scene-layer clip" data-composition-id="{h["data-composition-id"]}"\n'
                f'      data-composition-src="{h["data-composition-src"]}" data-timeline-label="{escape(h["id"])}"\n'
                f'      data-start="{st:.3f}" data-duration="{du:.3f}" data-track-index="{h.get("data-track-index", 2)}"\n'
                f'      data-width="{W}" data-height="{H}"></div>')
            mapa_slots[h["id"]] = {"slot": s["dir"], "t": s["t"], "src": h["data-composition-src"]}

    # overlays con composicion (CTA): misma pieza que se renderiza en green screen
    mapa_ov = []
    for k, ov in enumerate(edl.get("overlays") or []):
        if not ov.get("comp"):
            continue
        cf = (edit / ov["comp"]).resolve()
        shutil.copy2(cf, comps / cf.name)
        cid = f"ov-{k + 1:02d}"
        comp_id = "cta" if "cta" in cf.stem else cf.stem
        lines.append(
            f'    <div id="{cid}" class="scene-layer clip" data-composition-id="{comp_id}"\n'
            f'      data-composition-src="compositions/{cf.name}" data-timeline-label="Overlay {k + 1} · {cf.stem}"\n'
            f'      data-start="{ov["start_in_output"]:.3f}" data-duration="{ov["duration"]:.3f}" data-track-index="8"\n'
            f'      data-width="{W}" data-height="{H}"></div>')
        mapa_ov.append({"id": cid, "idx": k, "comp": ov["comp"]})

    # audio (musica + SFX) con la ganancia real del loudnorm
    mapa_audio = None
    if spec:
        base = spec_path.parent
        speed = float(spec.get("speed", 1.0))
        try:
            g = medir_ganancia(cargar_partes(spec, base), speed,
                               spec.get("loudnorm", {"I": -14, "TP": -1, "LRA": 11}))
        except (SystemExit, Exception) as e:  # partes aun sin renderizar
            print(f"  aviso: sin partes renderizadas para medir la ganancia ({e}); se usa +5 dB")
            g = 5.0
        aud, mapa_audio = clips_audio(spec, base, assets, g, speed, track0=10)
        lines += aud

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width={W}, height={H}"/>
  <title>Studio maestro</title>
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
  <style>
    html, body {{ margin: 0; width: {W}px; height: {H}px; overflow: hidden; background: #000; }}
    #root {{ position: relative; width: {W}px; height: {H}px; overflow: hidden; }}
    #root video {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: fill; }}
    .scene-layer {{ position: absolute; top: 0; left: 0; width: {W}px; height: {H}px; }}
    .blur-censura {{ backdrop-filter: blur(16px); background: rgba(239, 68, 68, 0.10);
                    outline: 3px dashed rgba(239, 68, 68, 0.9); }}
  </style>
</head>
<body>
  <!-- Generado por video-use/helpers/studio_maestro.py: el Studio UNICO del video (regla de oro).
       Cortes, encuadres, blurs, tarjetas, CTA, musica y SFX se editan aqui; from-studio los
       devuelve al pipeline. A 1x (el entregable va a {spec.get("speed", 1) if spec else 1}x). -->
  <div id="root" data-composition-id="maestro" data-start="0" data-duration="{total:.3f}"
    data-width="{W}" data-height="{H}">
{chr(10).join(lines)}
  </div>
  <script>
    const tl = gsap.timeline({{ paused: true }});
    tl.to({{}}, {{ duration: {total:.3f} }}, 0);
    window.__timelines["maestro"] = tl;
  </script>
</body>
</html>
"""
    index.write_text(html, encoding="utf-8")
    stamp.write_text(sha(html))
    mapa = {"ranges": edl["ranges"], "dims": dims, "slots": mapa_slots, "overlays": mapa_ov,
            "audio": mapa_audio, "total": total}
    (out / "maestro_map.json").write_text(json.dumps(mapa, ensure_ascii=False, indent=2), encoding="utf-8")
    pkg = load_json(PRESET / "package.json")
    pkg["name"] = "studio"
    (out / "package.json").write_text(json.dumps(pkg, indent=2), encoding="utf-8")
    shutil.copy(PRESET / "hyperframes.json", out / "hyperframes.json")
    n_rf = sum(1 for r in edl["ranges"] if r.get("reframe"))
    print(f"studio maestro: {len(edl['ranges'])} cortes ({n_rf} con encuadre), {nb} blurs, "
          f"{len(mapa_slots)} sub-comps de slots, {len(mapa_ov)} overlays, "
          f"{len(mapa_audio['music']) if mapa_audio else 0} musica + {len(mapa_audio['sfx']) if mapa_audio else 0} SFX"
          f" | {fmt_t(total)} -> {index}")


# ── from-studio ───────────────────────────────────────────────────────────────

def from_studio(edit: Path, dry: bool) -> None:
    out = edit / DIR
    mapa = load_json(out / "maestro_map.json")
    base_ranges, dims = mapa["ranges"], {k: tuple(v) for k, v in mapa["dims"].items()}
    p = Elementos()
    p.feed((out / "index.html").read_text(encoding="utf-8"))
    por_id = {a["id"]: (tag, a) for tag, a in p.el}

    # 1) clips -> ranges, en el orden de Studio; su ventana en Studio y en la SALIDA
    clips = []
    for tag, a in p.el:
        if tag == "video" and "data-range" in a and not oculto(a):
            clips.append(a)
    clips.sort(key=lambda a: (float(a.get("data-start", 0)), a["_order"]))
    ranges, wins, notas, t_out = [], [], [], 0.0
    for a in clips:
        orig = base_ranges[int(a["data-range"])]
        r = {k: v for k, v in orig.items() if k not in ("censor",)}
        ms, du = float(a.get("data-media-start", 0)), float(a["data-duration"])
        r["start"], r["end"] = round(ms, 3), round(ms + du, 3)
        sw, sh = dims[r["source"]]
        place = place_de_css(a.get("style", ""), sw, sh)
        if place and place.get("oculto"):
            notas.append(f"  {a['id']}: el clip quedo fuera del lienzo -> se descarta")
            continue
        p_orig = place_de_reframe(orig.get("reframe"), sw, sh)
        if place is None:
            r.pop("reframe", None)
        elif not (p_orig and all(abs(x - y) < 2 for x, y in zip(place["src"] + place["dst"],
                                                               p_orig["src"] + p_orig["dst"]))):
            r["reframe"] = reframe_de_place(place)
        if (r.get("reframe") or None) != (orig.get("reframe") or None):
            notas.append(f"  {a['id']}: encuadre {'quitado' if 'reframe' not in r else 'nuevo ' + str(r['reframe']['src_crop'])}")
        st = float(a.get("data-start", 0))
        wins.append((st, st + du, t_out, place, r))
        ranges.append(r)
        t_out += du

    def a_salida(ts: float) -> float:
        """Tiempo de Studio -> tiempo de salida (sin los huecos que dejan los recortes)."""
        for s0, s1, o0, _, _ in wins:
            if ts < s1:
                return o0 + max(0.0, ts - s0)
        return t_out

    # 2) blurs -> censor de cada range que tocan
    for tag, a in p.el:
        if "blur-censura" not in (a.get("class") or "") or oculto(a):
            continue
        b0 = float(a["data-start"])
        b1 = b0 + float(a["data-duration"])
        rect = rect_div(a.get("style", ""))
        for s0, s1, _, place, r in wins:
            if s1 > b0 and s0 < b1:
                sw, sh = dims[r["source"]]
                x, y, wd, ht = lienzo_a_fuente(place, rect, sw, sh)
                x0, y0 = max(0, int(x)), max(0, int(y))
                box = {"x": x0, "y": y0, "w": int(min(sw, x + wd) - x0), "h": int(min(sh, y + ht) - y0)}
                if box["w"] > 2 and box["h"] > 2:
                    r.setdefault("censor", []).append(box)
    n_cens = sum(len(r.get("censor", [])) for r in ranges)
    n_cens0 = sum(len(r.get("censor") or []) for r in base_ranges)
    if n_cens != n_cens0:
        notas.append(f"  blurs: {n_cens0} -> {n_cens} cajas de censura")

    # 3) slots: tiempos de sus hosts (relativos al slot) y archivos de composicion
    cambios_slot: dict[str, list] = {}
    for hid, info in mapa["slots"].items():
        tag_a = por_id.get(hid)
        if not tag_a or oculto(tag_a[1]):
            notas.append(f"  {hid}: quitado del timeline (se conserva en el slot; borrarlo alla si era la idea)")
            continue
        a = tag_a[1]
        s0 = float(a["data-start"])
        o0, o1 = a_salida(s0), a_salida(s0 + float(a["data-duration"]))
        cambios_slot.setdefault(info["slot"], []).append((hid, round(o0 - info["t"], 3), round(o1 - o0, 3), info["src"]))

    # 4) overlays (CTA): inicio en la salida
    edl = load_json(edit / "edl.json")
    overlays = edl.get("overlays") or []
    for info in mapa["overlays"]:
        tag_a = por_id.get(info["id"])
        ov = overlays[info["idx"]]
        if not tag_a or oculto(tag_a[1]):
            ov["_quitar"] = True
            notas.append(f"  {info['id']}: overlay eliminado")
            continue
        nuevo = round(a_salida(float(tag_a[1]["data-start"])), 3)
        if abs(nuevo - ov["start_in_output"]) > 0.05:
            notas.append(f"  {info['id']}: start {ov['start_in_output']} -> {nuevo}")
        ov["start_in_output"] = nuevo
    overlays = [o for o in overlays if not o.pop("_quitar", False)]

    # 5) audio -> stitch.json (tiempos de Studio convertidos a la salida)
    spec_path = edit / "stitch" / "stitch.json"
    music = sfx = cambios_audio = None
    if mapa["audio"] and spec_path.exists():
        spec = load_json(spec_path)
        audios = {}
        for tag, a in p.el:
            if tag == "audio":
                b = dict(a)
                if "data-start" in b:
                    s0 = float(b["data-start"])
                    o0 = a_salida(s0)
                    b["data-start"] = str(o0)
                    if "data-duration" in b:
                        b["data-duration"] = str(a_salida(s0 + float(a["data-duration"])) - o0)
                audios[a["id"]] = b
        music, sfx, cambios_audio = leer_audio(spec, mapa["audio"], audios, 1.0)
        notas += cambios_audio

    n0, n1 = len(base_ranges), len(ranges)
    print(f"cortes: {n0} -> {n1} | salida {fmt_t(sum(r['end'] - r['start'] for r in base_ranges))} -> {fmt_t(t_out)}")
    print("\n".join(notas) if notas else "  sin cambios de encuadre, blurs, overlays ni mezcla")
    if dry:
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bk = edit / "edl_backups"
    bk.mkdir(exist_ok=True)
    shutil.copy2(edit / "edl.json", bk / f"edl_{stamp}.json")
    edl["ranges"], edl["overlays"] = ranges, overlays
    (edit / "edl.json").write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
    # composiciones editadas en Studio -> de vuelta a su slot / overlay
    for sdir, hs in cambios_slot.items():
        sp = (edit / sdir).resolve()
        idx = sp / "index.html"
        shutil.copy2(idx, bk / f"{sp.name}_index_{stamp}.html")
        txt = idx.read_text(encoding="utf-8")

        for hid, st, du, src in hs:
            txt = re.sub(rf'(id="{re.escape(hid)}"[^>]*?data-start=")[\d.]+(")', rf"\g<1>{st:.3f}\g<2>", txt, count=1)
            txt = re.sub(rf'(id="{re.escape(hid)}"[^>]*?data-duration=")[\d.]+(")', rf"\g<1>{du:.3f}\g<2>", txt, count=1)
            f = out / src
            if f.exists() and f.read_bytes() != (sp / src).read_bytes():
                shutil.copy2(f, sp / src)
                print(f"  {src}: editada en Studio -> copiada a {sdir}")
        idx.write_text(txt, encoding="utf-8")
    for info in mapa["overlays"]:
        f, dst = out / "compositions" / Path(info["comp"]).name, (edit / info["comp"]).resolve()
        if f.exists() and dst.exists() and f.read_bytes() != dst.read_bytes():
            shutil.copy2(f, dst)
            print(f"  {info['comp']}: editada en Studio -> hay que re-renderizar el overlay")
    if music is not None and cambios_audio:
        spec = load_json(spec_path)
        shutil.copy2(spec_path, bk / f"stitch_{stamp}.json")
        spec["music"], spec["sfx"] = music, sfx
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  escrito edl.json (respaldos en edl_backups/, {stamp})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["to-studio", "from-studio"])
    ap.add_argument("--edit-dir", required=True, type=Path)
    ap.add_argument("--force", action="store_true", help="to-studio: descartar cambios sin exportar")
    ap.add_argument("--dry-run", action="store_true", help="from-studio: solo reportar")
    args = ap.parse_args()
    edit = args.edit_dir.resolve()
    if args.cmd == "to-studio":
        to_studio(edit, args.force)
    else:
        from_studio(edit, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
