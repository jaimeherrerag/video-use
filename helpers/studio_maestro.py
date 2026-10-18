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
  B-roll      (shorts) `edl.broll`: clips de otra fuente encima de la camara, con su encuadre;
              se recortan, mueven y reencuadran igual que los cortes
  sujeto      (shorts) `ranges[i].sujeto`: la persona recortada encima de su clip (texto detras
              de la cabeza), con fondo liso opcional (recorte_color.py)
  color       (shorts) `edl.color`: grade de cada clip y transiciones animadas (cortinilla,
              fundido, corte) como clips "Color" que se mueven en el timeline (recorte_color.py)
  zoom        (shorts) `ranges[i].zoom`: push-in / pull-out animado de la camara del clip, con una
              marca "Zoom" en el timeline (zooms_studio.py)

Shorts y otros lienzos (desde 2026-10-08): `edl.studio.canvas` = {w, h} (default 1920x1080) y
`edl.studio.proxy: false` usa copias a resolucion completa (GOP 6) en vez del proxy 540p: un
short es corto y el maestro se renderiza tal cual (`edl.studio.render: "maestro"`, build_final).
Capas: z-index por CSS (camara y color 1 < B-roll y fondo del sujeto 2 < tarjetas 3 < sujeto 4
< cortinilla 5 < blur 6 < overlays 7); un host de slot elige la suya con `data-z` (texto detras
de la cabeza: 3).

    to-studio   --edit-dir E [--force]   EDL + stitch.json + slots -> E/studio/
    from-studio --edit-dir E [--dry-run] Studio -> edl.json, stitch.json, slots (con respaldos)

Studio reproduce a 1x (el entregable va a `speed`). Recortar un clip en Studio deja hueco
(no hace ripple): todo tiempo de Studio se convierte al timeline de SALIDA por el clip que
tiene debajo, asi el CTA, las tarjetas y los SFX siguen al contenido.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import json
import shutil
import subprocess
import sys
from datetime import datetime
from html import escape
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from geometria_studio import (W, H, css_de_place, fuente_a_lienzo, lienzo_a_fuente,  # noqa: E402
                              place_de_css, place_de_reframe, rect_div, reframe_de_place)
from mezcla_studio import clips_audio, leer_audio  # noqa: E402
from recorte_color import (asset_sujeto, attr_grading, es_mono, css_capas, html_color, html_sujetos,  # noqa: E402
                           js_transiciones, leer_color, leer_sujetos, plan_color, ventana_sujeto)
from zooms_studio import CSS as CSS_ZOOM, JS as JS_ZOOM, html_zooms, leer_zooms  # noqa: E402
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


def huella(f: Path) -> str:
    """Huella del contenido SIN los data-hf-id: Studio se los agrega a toda composicion que abre,
    y eso no es una edicion (antes se copiaba de vuelta y pisaba los cambios del pipeline)."""
    txt = re.sub(r'\s+data-hf-id="[^"]*"', "", f.read_text(encoding="utf-8"))
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()


def editada_en_studio(f: Path, mapa: dict, otra: Path) -> bool:
    """La composicion f (copia de Studio) cambio DENTRO de Studio desde el to-studio?
    Se compara contra la huella guardada al generar; asi un cambio hecho por el pipeline en el
    original (p.ej. el karaoke regenerado) no se pisa con la copia vieja de Studio."""
    if not f.exists():
        return False
    h0 = (mapa.get("comp_sha") or {}).get(f.name)
    if h0 is None:                                  # maestro generado antes de 2026-10-08
        return otra.exists() and huella(f) != huella(otra)
    return huella(f) != h0


def renombrar_comp(txt: str, viejo: str, nuevo: str) -> str:
    """Cambia el id de composicion de una sub-composicion (template, root, CSS del root y timeline)."""
    v = re.escape(viejo)
    txt = re.sub(rf'id="{v}-template"', f'id="{nuevo}-template"', txt)
    txt = re.sub(rf'(id|data-composition-id)="{v}"', rf'\g<1>="{nuevo}"', txt)
    txt = re.sub(rf"#{v}(?=[\s{{.:,>\[])", f"#{nuevo}", txt)
    return re.sub(rf"""__timelines\[(["']){v}\1\]""", f'__timelines["{nuevo}"]', txt)


def etiqueta(texto: str) -> str:
    """Texto de data-timeline-label, recortado y SIN < ni >: Studio reescribe el archivo con esos
    caracteres sin escapar y el parser de tiempos de HyperFrames corta el tag en el ">" (visto
    2026-10-08: el clip quedaba a pantalla completa, sin estilo ni ventana de tiempo)."""
    texto = texto.replace("->", "→").replace("<", "‹").replace(">", "›")
    return texto if len(texto) <= LABEL_MAX else texto[:LABEL_MAX - 1].rstrip() + "…"


def lienzo(edl_o_mapa: dict) -> tuple[int, int]:
    """Tamano del lienzo del maestro: edl.studio.canvas (shorts 1080x1920) o el 1920x1080 de siempre."""
    c = (edl_o_mapa.get("studio") or {}).get("canvas") or edl_o_mapa.get("canvas") or {}
    return int(c.get("w", W)), int(c.get("h", H))


def asset_completo(src: Path, dst: Path) -> None:
    """Copia a resolucion completa para un maestro sin proxy (shorts): GOP de 6 para que
    HyperFrames busque exacto al renderizar, y el audio del mic (0:a:0, el mismo de render.py)."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    print(f"  asset completo: {src.name} -> {dst.name}", flush=True)
    tmp = dst.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-g", "6", "-keyint_min", "6", "-sc_threshold", "0",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(dst)


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

    st_cfg = edl.get("studio") or {}
    CW, CH = lienzo(edl)
    con_proxy = st_cfg.get("proxy", True)

    # medios: proxy 540p (long-form; se reutiliza el de la revision de cortes si existe) o copia a
    # resolucion completa (shorts), y tamano real de cada fuente
    proxies, dims, fuentes = {}, {}, {}
    for key, src in edl["sources"].items():
        sp = Path(src) if Path(src).is_absolute() else (edit / src).resolve()
        fuentes[key] = sp
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        if con_proxy:
            dst = assets / f"proxy_{safe}.mp4"
            prev = edit / "cortes" / "assets" / dst.name
            if not dst.exists() and prev.exists():
                shutil.copy2(prev, dst)
            make_proxy(sp, dst)
        else:
            dst = assets / f"full_{safe}.mp4"
            asset_completo(sp, dst)
        proxies[key], dims[key] = dst.name, probe_wh(sp)

    # recorte de sujeto y color: en un maestro que se renderiza tal cual (shorts) va todo; en los largos
    # (proxy + render.py por tramos, 2026-10-18) render.py compone el recorte y aplica solo el look
    # blanco y negro, asi que cualquier otro look se rechaza aqui en vez de perderse en el render
    color = edl.get("color")
    largo = con_proxy or st_cfg.get("render") != "maestro"
    if color and largo:
        otros = [n for n, lk in (color.get("looks") or {}).items() if not es_mono(lk)]
        if otros:
            sys.exit(f"edl.color: en un largo render.py solo aplica looks blanco y negro (preset mono*); "
                     f"no sabe hacer: {', '.join(otros)}")

    # ventana de cada clip en la salida, y el plan de color sobre ellas (cambios anclados en la fuente)
    ventanas, t = [], 0.0
    for r in edl["ranges"]:
        d = round(r["end"] - r["start"], 3)
        ventanas.append((t, round(t + d, 3), r))
        t = round(t + d, 3)
    total = t
    looks = (color or {}).get("looks") or {}
    fijos, trans, look_en = plan_color(color, ventanas)

    lines, cache = [], {}
    clip_win, css_clip = [], []                     # (t0, t1, place) y el CSS de encuadre por range
    for i, (t0, t1, r) in enumerate(ventanas):
        d = round(t1 - t0, 3)
        sw, sh = dims[r["source"]]
        place = place_de_reframe(r.get("reframe"), sw, sh, CW, CH)
        css = css_de_place(place, sw, sh)
        text = range_words(edit / "transcripts", r["source"], r["start"], r["end"], cache) or r.get("beat", "")
        label = etiqueta(f"{i + 1} · {text}")
        style = f' style="{css}"' if css else ""
        g = attr_grading(looks.get(fijos[i]) if fijos[i] else None)
        lines.append(
            f'    <video id="c{i + 1:03d}" src="assets/{proxies[r["source"]]}" data-range="{i}" data-zclip="{i}"'
            f' data-source="{escape(r["source"])}" data-timeline-label="{escape(label)}"\n'
            f'      data-start="{t0:.3f}" data-duration="{d:.3f}" data-media-start="{r["start"]:.3f}"'
            f' data-has-audio="true" data-track-index="0"{g}{style} playsinline></video>')
        clip_win.append((t0, t1, place))
        css_clip.append(css)
    # transiciones de color a mitad de clip: copias graduadas encima de la camara
    lines += html_color(trans, looks, proxies, css_clip, etiqueta)

    # blurs: un rectangulo por censor y por range, en coordenadas del lienzo
    nb = 0
    for i, r in enumerate(edl["ranges"]):
        for c in r.get("censor") or []:
            t0, t1, place = clip_win[i]
            sw, sh = dims[r["source"]]
            x, y, wd, ht = fuente_a_lienzo(place, (c["x"], c["y"], c["w"], c["h"]), sw, sh, CW, CH)
            nb += 1
            lines.append(
                f'    <div id="blur-{nb:02d}" class="clip blur-censura" data-timeline-label="Blur {nb} (censura)"\n'
                f'      data-start="{t0:.3f}" data-duration="{t1 - t0:.3f}" data-track-index="1"\n'
                f'      style="position: absolute; left: {x:.1f}px; top: {y:.1f}px; width: {wd:.1f}px; '
                f'height: {ht:.1f}px;"></div>')

    # B-roll (shorts): clips de otra fuente encima de la camara, con su encuadre editable; si es de
    # la propia camara (toma mala, PiP) lleva el look de color de su momento
    fuentes_cam = {r["source"] for r in edl["ranges"]}
    for k, b in enumerate(edl.get("broll") or []):
        sw, sh = dims[b["source"]]
        css = css_de_place(place_de_reframe(b.get("reframe"), sw, sh, CW, CH), sw, sh)
        label = etiqueta(f"B-roll {k + 1} · {b.get('nota', b['source'])}")
        style = f' style="{css}"' if css else ""
        lines.append(
            f'    <video id="b{k + 1:02d}" src="assets/{proxies[b["source"]]}" data-broll="{k}"'
            f' data-source="{escape(b["source"])}" data-timeline-label="{escape(label)}"\n'
            f'      data-start="{b["at"]:.3f}" data-duration="{b["end"] - b["start"]:.3f}"'
            f' data-media-start="{b["start"]:.3f}" data-has-audio="false" data-track-index="1"'
            f'{attr_grading(looks.get(look_en(b["at"])) if b["source"] in fuentes_cam and look_en(b["at"]) else None)}'
            f'{style} muted playsinline></video>')

    # recorte de sujeto (shorts): la persona encima de su clip, con el look de color de su inicio
    sujetos, sujeto_assets = [], {}
    for i, (t0, t1, r) in enumerate(ventanas):
        w = ventana_sujeto(r)
        if not w:
            if r.get("sujeto"):
                print(f"  aviso: el sujeto del clip {i + 1} queda fuera del clip -> no se emite")
            continue
        a, b = w
        # sobre un fondo liso el borde lleva el color del sujeto; sobre la camara real, el de la camara
        # largos: el recorte sale de la fuente ORIGINAL bajada al ancho del lienzo (el proxy de 540p
        # no sirve para el render) y es el mismo archivo que compone render.py
        con_fondo = bool(r["sujeto"].get("fondo") or r["sujeto"].get("fondo_img"))
        if largo:
            nombre, span0 = asset_sujeto(edit, fuentes[r["source"]], a, b, assets, relleno=con_fondo, ancho=CW)
        else:
            nombre, span0 = asset_sujeto(edit, assets / proxies[r["source"]], a, b, assets, relleno=con_fondo)
        img = r["sujeto"].get("fondo_img")
        if img:
            ip = Path(img) if Path(img).is_absolute() else edit / img
            shutil.copy2(ip, assets / ip.name)
        st0, du = round(t0 + a - r["start"], 3), round(b - a, 3)
        if any(e["T"] < st0 + du and e["T"] + e["dur"] > st0 for e in trans):
            print(f"  aviso: el recorte del clip {i + 1} se cruza con una transicion de color: "
                  "queda con el look de su inicio (acotarlo con sujeto.src)")
        lk = look_en(st0)
        sujetos.append({"n": len(sujetos) + 1, "i": i, "t0": st0, "dur": du, "ms": round(a - span0, 3),
                        "asset": nombre, "fondo": r["sujeto"].get("fondo"), "fondo_img": r["sujeto"].get("fondo_img"),
                        "fondo_hasta": r["sujeto"].get("fondo_hasta"),
                        "fondo_hueco": r["sujeto"].get("fondo_hueco"), "css": css_clip[i],
                        "look": looks.get(lk) if lk else None})
        sujeto_assets[nombre] = {"span0": span0}
    lines += html_sujetos(sujetos, etiqueta)

    # zooms animados (shorts): una marca por zoom en el timeline; escalan la camara de su clip
    lineas_zoom, mapa_zoom = html_zooms(ventanas, etiqueta, (CW // 2, int(CH * 0.38)) if CW > CH else None)
    lines += lineas_zoom

    # slots HF (tarjetas, karaoke...): sus hosts en el timeline maestro
    # composiciones con el mismo nombre en dos slots (el karaoke de cada uno es subtitles.html con id
    # "subs") se renombran en el maestro; si no, la ultima pisa a las demas y sus timelines chocan
    mapa_slots = {}
    slots_cfg = (edl.get("studio") or {}).get("slots", [])
    nombres = [f.name for s in slots_cfg for f in ((edit / s["dir"]).resolve() / "compositions").glob("*.html")]
    for s in slots_cfg:
        sdir = (edit / s["dir"]).resolve()
        tag_slot = Path(s["dir"]).name
        renombre = {}                               # src del slot -> (src en el maestro, id viejo, id nuevo)
        for f in (sdir / "compositions").glob("*.html"):
            if nombres.count(f.name) > 1:
                txt = f.read_text(encoding="utf-8")
                m = re.search(r'data-composition-id="([^"]+)"', txt)
                viejo = m.group(1) if m else f.stem
                nuevo = f"{viejo}--{tag_slot}"
                dst = comps / f"{tag_slot}--{f.name}"
                dst.write_text(renombrar_comp(txt, viejo, nuevo), encoding="utf-8")
                renombre[f"compositions/{f.name}"] = (f"compositions/{dst.name}", viejo, nuevo)
            else:
                shutil.copy2(f, comps / f.name)
        for f in (sdir / "assets").iterdir():
            if f.is_file() and f.name not in ("base.mp4", "README.md"):
                shutil.copy2(f, assets / f.name)
        for h in hosts_de_slot(sdir / "index.html"):
            st, du = float(h["data-start"]) + s["t"], float(h["data-duration"])
            du = min(du, total - st)
            hid, cid, src = h["id"], h["data-composition-id"], h["data-composition-src"]
            if src in renombre:
                src, _, cid = renombre[src]
                hid = f"{h['id']}--{tag_slot}"
            lines.append(
                f'    <div id="{hid}" class="scene-layer clip" data-composition-id="{cid}"\n'
                f'      data-composition-src="{src}" data-timeline-label="{escape(hid)}"\n'
                f'      data-start="{st:.3f}" data-duration="{du:.3f}" data-track-index="{h.get("data-track-index", 2)}"\n'
                f'      data-width="{CW}" data-height="{CH}" style="z-index: {h.get("data-z", 3)};"></div>')
            mapa_slots[hid] = {"slot": s["dir"], "t": s["t"], "src": h["data-composition-src"], "hid": h["id"]}
            if hid != h["id"]:
                mapa_slots[hid].update({"src_maestro": src, "id_viejo": renombre[h["data-composition-src"]][1],
                                        "id_nuevo": cid})

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
            f'      data-width="{CW}" data-height="{CH}" style="z-index: 7;"></div>')
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
  <meta name="viewport" content="width={CW}, height={CH}"/>
  <title>Studio maestro</title>
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
  <style>
    html, body {{ margin: 0; width: {CW}px; height: {CH}px; overflow: hidden; background: #000; }}
    #root {{ position: relative; width: {CW}px; height: {CH}px; overflow: hidden; }}
    #root video {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: fill; }}
    /* capas por z-index (el track de Studio no ordena): camara 1, B-roll 2, tarjetas 3 (o su data-z),
       blur 6, overlays 7 */
    #root video[data-range] {{ z-index: 1; }}
    #root video[data-broll] {{ z-index: 2; }}
    .scene-layer {{ position: absolute; top: 0; left: 0; width: {CW}px; height: {CH}px; z-index: 3; }}
    .blur-censura {{ z-index: 6; backdrop-filter: blur(16px); background: rgba(239, 68, 68, 0.10);
                    outline: 3px dashed rgba(239, 68, 68, 0.9); }}
{css_capas(CW, CH) if (trans or sujetos) else ""}{CSS_ZOOM if mapa_zoom else ""}  </style>
</head>
<body>
  <!-- Generado por video-use/helpers/studio_maestro.py: el Studio UNICO del video (regla de oro).
       Cortes, encuadres, blurs, tarjetas, CTA, musica y SFX se editan aqui; from-studio los
       devuelve al pipeline. A 1x (el entregable va a {spec.get("speed", 1) if spec else 1}x). -->
  <div id="root" data-composition-id="maestro" data-start="0" data-duration="{total:.3f}"
    data-width="{CW}" data-height="{CH}">
{chr(10).join(lines)}
  </div>
  <script>
    const tl = gsap.timeline({{ paused: true }});
    tl.to({{}}, {{ duration: {total:.3f} }}, 0);
{js_transiciones(CW, CH) if trans else ""}{JS_ZOOM if mapa_zoom else ""}    window.__timelines["maestro"] = tl;
  </script>
</body>
</html>
"""
    index.write_text(html, encoding="utf-8")
    stamp.write_text(sha(html))
    mapa = {"ranges": edl["ranges"], "dims": dims, "slots": mapa_slots, "overlays": mapa_ov,
            "audio": mapa_audio, "total": total, "canvas": {"w": CW, "h": CH},
            "broll": edl.get("broll") or [],
            "color": color,
            "el_color": {f"cg{n:02d}a": {"start": e["copias"][0]["start"], "source": e["copias"][0]["source"],
                                          "k": e["k"]} for n, e in enumerate(trans, 1)},
            "sujetos": {f"s{s['n']:03d}": {"range": s["i"], "start": s["t0"], "dur": s["dur"], "ms": s["ms"]}
                        for s in sujetos},
            "sujeto_assets": sujeto_assets,
            "zooms": mapa_zoom,
            "comp_sha": {f.name: huella(f) for f in comps.glob("*.html")}}
    (out / "maestro_map.json").write_text(json.dumps(mapa, ensure_ascii=False, indent=2), encoding="utf-8")
    pkg = load_json(PRESET / "package.json")
    pkg["name"] = "studio"
    (out / "package.json").write_text(json.dumps(pkg, indent=2), encoding="utf-8")
    shutil.copy(PRESET / "hyperframes.json", out / "hyperframes.json")
    n_rf = sum(1 for r in edl["ranges"] if r.get("reframe"))
    print(f"studio maestro {CW}x{CH}: {len(edl['ranges'])} cortes ({n_rf} con encuadre), "
          f"{len(edl.get('broll') or [])} B-roll, {nb} blurs, {len(sujetos)} sujetos, {len(trans)} transiciones de color, "
          f"{len(mapa_zoom)} zooms, "
          f"{len(mapa_slots)} sub-comps de slots, {len(mapa_ov)} overlays, "
          f"{len(mapa_audio['music']) if mapa_audio else 0} musica + {len(mapa_audio['sfx']) if mapa_audio else 0} SFX"
          f" | {fmt_t(total)} -> {index}")


# ── from-studio ───────────────────────────────────────────────────────────────

def from_studio(edit: Path, dry: bool) -> None:
    out = edit / DIR
    mapa = load_json(out / "maestro_map.json")
    base_ranges, dims = mapa["ranges"], {k: tuple(v) for k, v in mapa["dims"].items()}
    CW, CH = lienzo(mapa)
    html_studio = (out / "index.html").read_text(encoding="utf-8")
    p = Elementos()
    p.feed(html_studio)
    por_id = {a["id"]: (tag, a) for tag, a in p.el}

    # 1) clips -> ranges, en el orden de Studio; su ventana en Studio y en la SALIDA
    clips = []
    for tag, a in p.el:
        if tag == "video" and "data-range" in a and not oculto(a):
            clips.append(a)
    clips.sort(key=lambda a: (float(a.get("data-start", 0)), a["_order"]))
    ranges, wins, notas, t_out = [], [], [], 0.0
    orig_idx, clip_attrs = [], []                   # por range nuevo: su range original y su <video>
    for a in clips:
        orig = base_ranges[int(a["data-range"])]
        r = {k: v for k, v in orig.items() if k not in ("censor", "sujeto", "zoom")}
        ms, du = float(a.get("data-media-start", 0)), float(a["data-duration"])
        r["start"], r["end"] = round(ms, 3), round(ms + du, 3)
        sw, sh = dims[r["source"]]
        place = place_de_css(a.get("style", ""), sw, sh, CW, CH)
        if place and place.get("oculto"):
            notas.append(f"  {a['id']}: el clip quedo fuera del lienzo -> se descarta")
            continue
        p_orig = place_de_reframe(orig.get("reframe"), sw, sh, CW, CH)
        if place is None:
            r.pop("reframe", None)
        elif not (p_orig and all(abs(x - y) < 2 for x, y in zip(place["src"] + place["dst"],
                                                               p_orig["src"] + p_orig["dst"]))):
            r["reframe"] = reframe_de_place(place, CW, CH)
            # Un arrastre accidental de pocos px deja una franja negra casi invisible en Studio
            # (en un short la camara ocupa solo la mitad de abajo: solo cuentan los bordes del lienzo
            # que el clip ya tocaba)
            dx, dy, dw, dh = place["dst"]
            ox, oy, ow, oh = p_orig["dst"] if p_orig else (0, 0, CW, CH)
            bordes = [n for n, g, toca in (("izq", dx, ox < 1), ("arriba", dy, oy < 1),
                                           ("der", CW - dx - dw, ox + ow > CW - 1),
                                           ("abajo", CH - dy - dh, oy + oh > CH - 1)) if toca and 0.5 < g < 12]
            if bordes:
                notas.append(f"  {a['id']}: OJO franja negra de pocos px ({', '.join(bordes)}) "
                             "-> ¿se movio sin querer?")
        if (r.get("reframe") or None) != (orig.get("reframe") or None):
            notas.append(f"  {a['id']}: encuadre {'quitado' if 'reframe' not in r else 'nuevo ' + str(r['reframe']['src_crop'])}")
        st = float(a.get("data-start", 0))
        wins.append((st, st + du, t_out, place, r))
        ranges.append(r)
        orig_idx.append(int(a["data-range"]))
        clip_attrs.append(a)
        t_out += du

    def a_salida(ts: float) -> float:
        """Tiempo de Studio -> tiempo de salida (sin los huecos que dejan los recortes)."""
        for s0, s1, o0, _, _ in wins:
            if ts < s1:
                return o0 + max(0.0, ts - s0)
        return t_out

    # 1b) recorte de sujeto (sigue a su clip) y 1c) color (anclas de las transiciones + grade de cada clip)
    notas += leer_sujetos(p.el, ranges, orig_idx, base_ranges, mapa, oculto)
    notas += leer_zooms(p.el, wins, ranges, orig_idx, base_ranges, mapa, oculto)
    color_nuevo, notas_color = leer_color(p.el, wins, clip_attrs, mapa, a_salida, oculto)
    notas += notas_color

    # 2) blurs -> censor de cada range que tocan
    for tag, a in p.el:
        if "blur-censura" not in (a.get("class") or "") or oculto(a):
            continue
        b0 = float(a["data-start"])
        b1 = b0 + float(a["data-duration"])
        rect = rect_div(a.get("style", ""))
        for s0, s1, _, place, r in wins:
            # un blur que toca al clip vecino por el redondeo de Studio (1-8 ms) no es suyo
            if min(s1, b1) - max(s0, b0) > 0.02:
                sw, sh = dims[r["source"]]
                x, y, wd, ht = lienzo_a_fuente(place, rect, sw, sh, CW, CH)
                x0, y0 = max(0, int(x)), max(0, int(y))
                box = {"x": x0, "y": y0, "w": int(min(sw, x + wd) - x0), "h": int(min(sh, y + ht) - y0)}
                if box["w"] > 2 and box["h"] > 2:
                    r.setdefault("censor", []).append(box)
    n_cens = sum(len(r.get("censor", [])) for r in ranges)
    n_cens0 = sum(len(r.get("censor") or []) for r in base_ranges)
    if n_cens != n_cens0:
        notas.append(f"  blurs: {n_cens0} -> {n_cens} cajas de censura")

    # 2b) B-roll (shorts): inicio en la salida, tramo de su fuente y encuadre
    broll = []
    base_broll = mapa.get("broll") or []
    vistos = set()
    for tag, a in p.el:
        if tag != "video" or "data-broll" not in a or oculto(a):
            continue
        k = int(a["data-broll"])
        orig = base_broll[k]
        b = dict(orig)
        sw, sh = dims[b["source"]]
        s0, du = float(a.get("data-start", 0)), float(a["data-duration"])
        o0, o1 = a_salida(s0), a_salida(s0 + du)
        ms = float(a.get("data-media-start", 0))
        b["at"], b["start"], b["end"] = round(o0, 3), round(ms, 3), round(ms + (o1 - o0), 3)
        place = place_de_css(a.get("style", ""), sw, sh, CW, CH)
        if place and place.get("oculto"):
            notas.append(f"  {a['id']}: B-roll fuera del lienzo -> se descarta")
            continue
        p_orig = place_de_reframe(orig.get("reframe"), sw, sh, CW, CH)
        if place is None:
            b.pop("reframe", None)
        elif not (p_orig and all(abs(x - y) < 2 for x, y in zip(place["src"] + place["dst"],
                                                               p_orig["src"] + p_orig["dst"]))):
            b["reframe"] = reframe_de_place(place, CW, CH)
        cambios_b = [c for c in ("at", "start", "end") if abs(b[c] - orig[c]) > 0.02]
        if b.get("reframe") != orig.get("reframe"):
            cambios_b.append("encuadre")
        if k in vistos:
            cambios_b.append("partido")
        vistos.add(k)
        if cambios_b:
            notas.append(f"  {a['id']}: B-roll {', '.join(cambios_b)}")
        broll.append(b)
    for k in range(len(base_broll)):
        if k not in vistos:
            notas.append(f"  b{k + 1:02d}: B-roll eliminado")
    broll.sort(key=lambda b: b["at"])

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
        # lo que llegaba al final del video (fondo, karaoke, tarjeta de cierre) sigue al nuevo final
        # si Jaime alargo o recorto el ultimo clip
        if abs(s0 + float(a["data-duration"]) - mapa["total"]) < 0.05 and abs(t_out - mapa["total"]) > 0.02:
            o1 = t_out
            notas.append(f"  {hid}: termina con el video ({mapa['total']:.2f} -> {t_out:.2f})")
        cambios_slot.setdefault(info["slot"], []).append((info.get("hid", hid), round(o0 - info["t"], 3),
                                                          round(o1 - o0, 3), info["src"], info))

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
    if base_broll or broll:
        edl["broll"] = broll
    if color_nuevo is not None:
        edl["color"] = color_nuevo
    (edit / "edl.json").write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
    # composiciones editadas en Studio -> de vuelta a su slot / overlay
    for sdir, hs in cambios_slot.items():
        sp = (edit / sdir).resolve()
        idx = sp / "index.html"
        shutil.copy2(idx, bk / f"{sp.name}_index_{stamp}.html")
        txt = idx.read_text(encoding="utf-8")

        for hid, st, du, src, info in hs:
            txt = re.sub(rf'(id="{re.escape(hid)}"[^>]*?data-start=")[\d.]+(")', rf"\g<1>{st:.3f}\g<2>", txt, count=1)
            txt = re.sub(rf'(id="{re.escape(hid)}"[^>]*?data-duration=")[\d.]+(")', rf"\g<1>{du:.3f}\g<2>", txt, count=1)
            f = out / info.get("src_maestro", src)
            if editada_en_studio(f, mapa, sp / src):
                shutil.copy2(sp / src, bk / f"{Path(src).stem}_{stamp}.html")
                if info.get("src_maestro"):         # vuelve con el id que espera el slot
                    (sp / src).write_text(renombrar_comp(f.read_text(encoding="utf-8"), info["id_nuevo"],
                                                         info["id_viejo"]), encoding="utf-8")
                else:
                    shutil.copy2(f, sp / src)
                print(f"  {src}: editada en Studio -> copiada a {sdir}")
        idx.write_text(txt, encoding="utf-8")
    for info in mapa["overlays"]:
        f, dst = out / "compositions" / Path(info["comp"]).name, (edit / info["comp"]).resolve()
        if dst.exists() and editada_en_studio(f, mapa, dst):
            shutil.copy2(f, dst)
            print(f"  {info['comp']}: editada en Studio -> hay que re-renderizar el overlay")
    if music is not None and cambios_audio:
        spec = load_json(spec_path)
        shutil.copy2(spec_path, bk / f"stitch_{stamp}.json")
        spec["music"], spec["sfx"] = music, sfx
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    # este estado de Studio ya quedo exportado: to-studio puede regenerar sin --force (Studio
    # reescribe el archivo al abrirlo, p.ej. con data-hf-id, y eso no es una edicion pendiente)
    (out / ".generated.sha1").write_text(sha(html_studio))
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
