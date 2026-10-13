#!/usr/bin/env python3
"""recorte_color.py - recorte de sujeto y color animado del Studio maestro (biblioteca de studio_maestro).

Dos edits nuevos (2026-10-12), visibles y editables en Studio por la regla de oro:

  sujeto  `ranges[i].sujeto = {"fondo": "#2e4a2a" | null, "fondo_hueco": {"elipse": [cx, cy, rx, ry,
          nucleo], "piso": [y0, y1]}?, "fondo_hasta": [y0, y1]?, "src": [a, b]?}`:
          la persona recortada con
          `hyperframes remove-background` (webm VP9 con alfa) encima de su clip, para poner texto
          DETRAS de la cabeza (host de slot con data-z 3). Con `fondo`, un color liso tapa la camara;
          `fondo_hueco` deja ver la camara real en una elipse (manos y antebrazos, donde el recorte
          falla) y en una franja inferior (escritorio); `fondo_hasta` solo desvanece hacia abajo.
          Es propiedad del clip, asi que sigue sus recortes y partidas en Studio. `src` (tiempo de la
          fuente) limita la ventana; sin `src` cubre el clip entero. Los recortes se cachean en
          <edit>/sujetos/ (minutos de CPU) con +-0.5 s de margen.
  color   `edl.color = {"looks": {nombre: grading}, "inicial": nombre|null, "cambios": [
              {"source", "src", "look", "transicion": "cortinilla"|"fundido"|"corte", "dur"}]}`,
          anclados en el tiempo de la FUENTE para que sigan a los cortes. Cada clip lleva el
          data-color-grading del look que ya termino de entrar cuando empieza; una transicion a mitad
          de clip es una capa (.capa-color) con copias mudas, ya graduadas, de los clips que toca,
          revelada con clip-path (cortinilla, con linea y pastillas Antes/Despues) u opacidad
          (fundido, corte). Los tweens leen data-start del DOM: mover el clip "Color" en Studio
          mueve la transicion.

Probado 2026-10-12 en una composicion de prueba: el grade sobre el webm con alfa conserva el alfa,
la cortinilla leida del DOM funciona en el render y remove-background saca un keyframe por cuadro
(no hace falta recodificar con GOP 6).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Callable

HF = "0.8.140"
FPS = 30
MARGEN = 0.5                       # s de fuente de mas a cada lado del recorte (recortes en Studio)
MASCARA = 4                        # version del post-proceso del borde (_limpiar_mascara)
TRACK_FONDO, TRACK_SUJETO, TRACK_COLOR = 19, 20, 30
# cortinilla diagonal: la zona "despues" entra desde la derecha; el borde va inclinado 20% del ancho
POLY_INI = "polygon(120% 0%, 140% 0%, 140% 100%, 100% 100%)"
POLY_FIN = "polygon(0% 0%, 140% 0%, 140% 100%, -20% 100%)"


def look_json(look) -> str:
    """JSON canonico de un grading: misma salida siempre (idempotencia de to-studio)."""
    return json.dumps(look, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def attr_grading(look: dict | None) -> str:
    return f' data-color-grading="{escape(look_json(look))}"' if look else ""


# ── recorte de sujeto ─────────────────────────────────────────────────────────

def _copiar(src: Path, dst: Path) -> None:
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)


def _tramo(full: Path, s0: float, s1: float, dst: Path) -> None:
    """El tramo [s0, s1] de la fuente del maestro, cuadro por cuadro (entrada de remove-background y
    color del recorte)."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s0:.3f}", "-i", str(full), "-t", f"{s1 - s0:.3f}",
                    "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-g", "6", "-r", str(FPS),
                    str(dst)], check=True)


def _limpiar_mascara(tramo: Path, crudo: Path, dst: Path, relleno: bool = True) -> None:
    """Borde limpio del recorte (version MASCARA), cuadro por cuadro:
    - alfa: el de u2net cerrado ~2 px hacia adentro, con la franja semitransparente reducida
      (25-75% -> 0-100%) y suavizada 0.6 px;
    - color: el EXACTO de la camara, salvo en el borde semitransparente y 3 px afuera, donde va el
      color del propio sujeto extendido hacia afuera (convolucion normalizada), no el de la pared. Mas
      lejos vuelve la camara: con color oscuro ahi, el alfa residual de la compresion (~2/255 de media,
      picos en pelos finos) dejaba una sombra alrededor del pelo en el render (v3, 2026-10-13).
      Con `relleno=False` (recorte sobre la camara real) el color es el de la camara en todo el cuadro:
      ahi compone casi identico a la camara (0.9 niveles de diferencia en el borde, contra 6.8 con
      relleno), y el relleno solo sirve sobre un fondo liso.
    Por que (2026-10-12/13): con el color de la pared (lila casi blanco) en los pixeles transparentes,
    el Studio de Jaime pintaba una linea blanca de ~3 px alrededor de los recortes graduados, y sobre
    el fondo liso quedaba un halo lila. En el render no se veia, pero un visor que mezcla vecinos del
    borde (al escalar, al subir la textura) jala ese color."""
    import numpy as np
    from scipy import ndimage

    w, h = (int(x) for x in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(tramo)], capture_output=True, text=True, check=True).stdout.strip().split(","))
    dec_c = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(tramo), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             stdout=subprocess.PIPE)
    dec_a = subprocess.Popen(["ffmpeg", "-v", "error", "-c:v", "libvpx-vp9", "-i", str(crudo), "-vf", "alphaextract",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-crf", "18",
                            "-b:v", "0", "-deadline", "good", "-cpu-used", "4", "-row-mt", "1", "-g", "1",
                            "-auto-alt-ref", "0", str(dst)], stdin=subprocess.PIPE)
    n = 0
    while True:
        bc, ba = dec_c.stdout.read(w * h * 3), dec_a.stdout.read(w * h)
        if len(bc) < w * h * 3 or len(ba) < w * h:
            break
        c = np.frombuffer(bc, np.uint8).reshape(h, w, 3).astype(np.float32)
        a = np.frombuffer(ba, np.uint8).reshape(h, w).astype(np.float32) / 255
        a = ndimage.grey_erosion(a, size=(5, 5))
        a = ndimage.gaussian_filter(np.clip((a - 0.25) / 0.5, 0, 1), 0.6)
        solido = (a >= 0.98).astype(np.float32)
        rgb = c.copy()
        # solo la franja de 3 px pegada al borde lleva el color del sujeto (la que un visor mezcla al
        # escalar); mas afuera queda la camara, asi el alfa residual de la compresion no ensucia el render
        hueco = ndimage.binary_dilation(a > 0.02, iterations=3) & (solido < 1)
        if not relleno:
            hueco[:] = False
        for tam in (15, 61):                      # primero el color cercano, luego el lejano
            den = ndimage.uniform_filter(solido, tam)
            num = np.stack([ndimage.uniform_filter(c[..., k] * solido, tam) for k in range(3)], -1)
            ok = hueco & (den > 1e-3)
            rgb[ok] = num[ok] / den[ok][:, None]
            hueco &= ~ok
        if hueco.any():
            rgb[hueco] = c[solido > 0].mean(0) if solido.any() else c[hueco]
        enc.stdin.write(np.dstack([rgb, a * 255]).clip(0, 255).astype(np.uint8).tobytes())
        n += 1
    enc.stdin.close()
    for pr in (dec_c, dec_a, enc):
        pr.wait()
    if enc.returncode != 0 or n == 0:
        sys.exit(f"limpieza del borde fallo ({n} cuadros)")


def asset_sujeto(edit: Path, full: Path, a: float, b: float, assets: Path,
                 relleno: bool = True) -> tuple[str, float]:
    """Recorte (webm con alfa) que cubre [a, b] de la fuente `full` (la misma copia que usa la camara
    del maestro, asi los cuadros coinciden). Reusa uno de la cache si su tramo lo contiene; si no,
    genera [a - 0.5, b + 0.5] alineado a cuadros. En la cache quedan el crudo de remove-background
    (lo caro) y su version con el borde limpio (`_m<MASCARA>`), que es la que va a Studio.
    Devuelve (archivo en assets/, inicio del tramo)."""
    cache = edit / "sujetos"
    cache.mkdir(exist_ok=True)
    idx_p = cache / "indice.json"
    idx = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {}
    st = full.stat()
    firma = {"size": st.st_size, "mtime": int(st.st_mtime)}
    crudo = s0 = s1 = None
    for nombre, info in sorted(idx.items()):
        c0, c1 = info["span"]
        if (info["fuente"] == full.name and info["firma"] == firma and c0 <= a + 1e-3 and b <= c1 + 1e-3
                and (cache / nombre).exists()):
            crudo, s0, s1 = cache / nombre, c0, c1
            break
    if crudo is None:
        s0 = max(0.0, math.floor((a - MARGEN) * FPS) / FPS)
        s1 = math.ceil((b + MARGEN) * FPS) / FPS
        h = hashlib.sha1(f"{full.name}|{firma}|{s0:.3f}|{s1:.3f}|best|{HF}".encode()).hexdigest()[:8]
        crudo, tramo = cache / f"sujeto_{h}.webm", cache / f"_tramo_{h}.mp4"
        print(f"  recorte de sujeto: {full.name} {s0:.2f}-{s1:.2f} s "
              f"(remove-background en CPU, ~{(s1 - s0) * FPS * 0.3:.0f} s)...", flush=True)
        _tramo(full, s0, s1, tramo)
        npx = ["cmd", "/c", "npx"] if sys.platform == "win32" else ["npx"]
        # su spinner llena la consola: se captura y solo se muestra la ultima linea (o todo si falla)
        rb = subprocess.run(npx + ["--yes", f"hyperframes@{HF}", "remove-background", str(tramo),
                                   "-o", str(crudo), "--quality", "best"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
        salida = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "\n", (rb.stdout or "") + (rb.stderr or ""))
        lineas = [x.strip(" │◇◒◐◓◑") for x in salida.splitlines() if x.strip(" │◇◒◐◓◑")]
        if rb.returncode != 0 or not crudo.exists():
            sys.exit("remove-background fallo:\n" + "\n".join(lineas[-15:]))
        print("   ", next((x for x in reversed(lineas) if "Removed background" in x), lineas[-1] if lineas else "ok"))
        idx[crudo.name] = {"fuente": full.name, "firma": firma, "span": [round(s0, 3), round(s1, 3)]}
        idx_p.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    limpio = crudo.with_name(f"{crudo.stem}_m{MASCARA}{'' if relleno else 'c'}.webm")
    if not limpio.exists():
        print(f"  borde del recorte: {limpio.name}", flush=True)
        tramo = cache / f"_tramo_{crudo.stem}.mp4"
        if not tramo.exists():
            _tramo(full, s0, s1, tramo)
        _limpiar_mascara(tramo, crudo, limpio, relleno)
        tramo.unlink(missing_ok=True)
    _copiar(limpio, assets / limpio.name)
    return limpio.name, s0


def ventana_sujeto(r: dict) -> tuple[float, float] | None:
    """[a, b] de la fuente donde va el recorte del range r (su `src` recortado al clip)."""
    s = r.get("sujeto")
    if not s:
        return None
    a, b = (s.get("src") or [r["start"], r["end"]])
    a, b = max(a, r["start"]), min(b, r["end"])
    return (a, b) if b - a >= 0.05 else None


def html_sujetos(sujetos: list[dict], etq: Callable[[str], str]) -> list[str]:
    """sujetos: [{n, i, t0, dur, ms, asset, fondo, css, look}] -> fondo liso (z2) + recorte (z4)."""
    lines = []
    for s in sujetos:
        n, style = s["n"], f' style="{s["css"]}"' if s["css"] else ""
        if s["fondo"]:
            mascara = ""
            hueco = s.get("fondo_hueco") or {}
            if hueco:
                # la camara real se ve en una elipse (manos y antebrazos, donde el recorte falla) y en una
                # franja inferior (escritorio); el fondo queda en la interseccion de las dos mascaras
                capas = []
                if hueco.get("elipse"):
                    cx, cy, rx, ry, nucleo = hueco["elipse"]
                    capas.append(f"radial-gradient(ellipse {rx}px {ry}px at {cx}px {cy}px, "
                                 f"transparent {nucleo * 100:.0f}%, #000 100%)")
                if hueco.get("piso"):
                    y0, y1 = hueco["piso"]
                    capas.append(f"linear-gradient(to bottom, #000 {y0}px, transparent {y1}px)")
                g = ", ".join(capas)
                mascara = (f" -webkit-mask-image: {g}; -webkit-mask-composite: source-in;"
                           f" mask-image: {g}; mask-composite: intersect;")
            elif s.get("fondo_hasta"):            # [y0, y1]: solido hasta y0, transparente desde y1
                y0, y1 = s["fondo_hasta"]
                g = f"linear-gradient(to bottom, #000 {y0}px, transparent {y1}px)"
                mascara = f" -webkit-mask-image: {g}; mask-image: {g};"
            lines.append(
                f'    <div id="fs{n:03d}" class="clip fondo-sujeto" data-sujeto="{s["i"]}"'
                f' data-timeline-label="{escape(etq(f"Fondo sujeto {n}"))}"\n'
                f'      data-start="{s["t0"]:.3f}" data-duration="{s["dur"]:.3f}" data-track-index="{TRACK_FONDO}"'
                f' style="background: {s["fondo"]};{mascara}"></div>')
        lines.append(
            f'    <div id="sw{n:03d}" class="capa-sujeto">\n'
            f'      <video id="s{n:03d}" src="assets/{s["asset"]}" data-sujeto="{s["i"]}" data-zclip="{s["i"]}"'
            f' data-timeline-label="{escape(etq(f"Sujeto {n} · recorte (clip {s["i"] + 1})"))}"\n'
            f'        data-start="{s["t0"]:.3f}" data-duration="{s["dur"]:.3f}" data-media-start="{s["ms"]:.3f}"'
            f' data-has-audio="false" data-track-index="{TRACK_SUJETO + n - 1}"{attr_grading(s["look"])}{style}'
            f' muted playsinline></video>\n'
            f'    </div>')
    return lines


# ── color ─────────────────────────────────────────────────────────────────────

def anclar(c: dict, clips: list) -> tuple[float | None, str]:
    """Tiempo de salida de un cambio anclado en la fuente. clips: [(t0, t1, range)]."""
    for t0, t1, r in clips:
        if r["source"] == c["source"] and r["start"] - 1e-3 <= c["src"] < r["end"]:
            return t0 + c["src"] - r["start"], ""
    sig = sorted((r["start"], t0) for t0, t1, r in clips if r["source"] == c["source"] and r["start"] > c["src"])
    if sig:
        return sig[0][1], f"cayo en un hueco del corte (src {c['src']}): va al inicio del clip siguiente"
    return None, f"cayo fuera del corte (src {c['src']}): se ignora"


def plan_color(color: dict | None, clips: list) -> tuple[list, list, Callable[[float], str | None]]:
    """-> (look fijo de cada clip, transiciones a mitad de clip con sus copias, look_en(t))."""
    if not color:
        return [None] * len(clips), [], lambda t: None
    looks = color.get("looks") or {}
    eventos = []
    for k, c in enumerate(color.get("cambios") or []):
        if c.get("look") is not None and c["look"] not in looks:
            sys.exit(f"edl.color: look desconocido '{c['look']}' (cambio {k})")
        T, aviso = anclar(c, clips)
        if aviso:
            print(f"  aviso color: cambio {k} ({c.get('look')}) {aviso}")
        if T is None:
            continue
        tipo = c.get("transicion", "corte")
        dur = 0.0 if tipo == "corte" else float(c.get("dur", 0.5))
        eventos.append({"k": k, "T": round(T, 3), "dur": dur, "look": c.get("look"), "tipo": tipo,
                        "source": c["source"]})
    eventos.sort(key=lambda e: (e["T"], e["k"]))
    inicial = color.get("inicial")

    def look_en(t: float):
        """Look que ya termino de entrar en t."""
        lk = inicial
        for e in eventos:
            if e["T"] + e["dur"] <= t + 0.02:
                lk = e["look"]
        return lk

    fijos = [look_en(t0) for t0, _, _ in clips]
    trans = []
    for e in eventos:
        T, fin = e["T"], e["T"] + max(e["dur"], 0.001)
        copias = []
        for i, (t0, t1, r) in enumerate(clips):
            if t1 <= T + 1e-3 or t0 >= fin:
                continue
            if e["tipo"] == "corte" and abs(t0 - T) <= 0.02:
                continue                            # corte justo en el borde: lo da el grade fijo
            a0 = max(T, t0)
            copias.append({"i": i, "start": round(a0, 3), "dur": round(t1 - a0, 3),
                           "ms": round(r["start"] + a0 - t0, 3), "source": r["source"]})
        if copias:
            trans.append({**e, "copias": copias})
    return fijos, trans, look_en


def html_color(trans: list, looks: dict, asset_de: dict, css: list, etq: Callable[[str], str]) -> list[str]:
    """Capas .capa-color (encima de la camara) y la UI de cada cortinilla."""
    lines = []
    for n, e in enumerate(trans, 1):
        g = attr_grading(looks.get(e["look"]) if e["look"] else None)
        wstyle = f"clip-path: {POLY_INI};" if e["tipo"] == "cortinilla" else "opacity: 0;"
        lines.append(f'    <div id="cw{n:02d}" class="capa-color" style="{wstyle}">')
        for j, c in enumerate(e["copias"]):
            ancla = (f' data-ancla="" data-transicion="{e["tipo"]}" data-trans-dur="{e["dur"]:.3f}"'
                     if j == 0 else "")
            nombre = e["look"] or "sin color"
            label = f"Color · {nombre} ({e['tipo']})" if j == 0 else f"Color · {nombre} (sigue)"
            style = f' style="{css[c["i"]]}"' if css[c["i"]] else ""
            lines.append(
                f'      <video id="cg{n:02d}{chr(97 + j)}" src="assets/{asset_de[c["source"]]}" data-color="{e["k"]}"'
                f' data-zclip="{c["i"]}"'
                f'{ancla} data-timeline-label="{escape(etq(label))}"\n'
                f'        data-start="{c["start"]:.3f}" data-duration="{c["dur"]:.3f}" data-media-start="{c["ms"]:.3f}"'
                f' data-has-audio="false" data-track-index="{TRACK_COLOR + n - 1}"{g}{style} muted playsinline></video>')
        lines.append("    </div>")
        if e["tipo"] == "cortinilla":
            lines.append(
                f'    <div id="cw{n:02d}-ui" class="cortinilla-ui" style="opacity: 0;">\n'
                f'      <div class="cw-pastilla cw-antes">Antes</div><div class="cw-pastilla cw-despues">Después</div>\n'
                f'      <div class="cw-linea"><div class="cw-asa">‹›</div></div>\n'
                f'    </div>')
    return lines


def css_capas(cw: int, ch: int) -> str:
    return f"""    /* recorte y color (recorte_color.py): capa-color va despues de la camara en el DOM */
    .capa-color, .fondo-sujeto, .capa-sujeto, .cortinilla-ui {{ position: absolute; inset: 0; }}
    .capa-color {{ z-index: 1; }}
    .fondo-sujeto {{ z-index: 2; }}
    .capa-sujeto {{ z-index: 4; }}
    .cortinilla-ui {{ z-index: 5; pointer-events: none; }}
    .cw-linea {{ position: absolute; left: 0; top: -60px; width: 6px; height: {ch + 120}px; margin-left: -3px;
                background: #efe7d4; }}
    .cw-asa {{ position: absolute; top: {ch // 2 + 26}px; left: -31px; width: 68px; height: 68px; border-radius: 50%;
              background: #efe7d4; color: #1a1a17; display: flex; align-items: center; justify-content: center;
              font-family: Consolas, "Courier New", monospace; font-size: 30px; font-weight: 500; }}
    .cw-pastilla {{ position: absolute; top: {330 if ch > cw else 96}px; padding: 12px 22px; border-radius: 6px; background: #1a1a17;
                   color: #efe7d4; font-family: Consolas, "Courier New", monospace; font-size: 26px;
                   font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; }}
    .cw-antes {{ left: 48px; }}
    .cw-despues {{ right: 48px; }}
"""


def js_transiciones(cw: int, ch: int) -> str:
    """Tweens de las transiciones de color, con los tiempos leidos del DOM."""
    rot = math.degrees(math.atan(0.2 * cw / ch))
    x0, x1 = 1.1 * cw, -0.1 * cw                  # centro de la linea: fuera a la derecha -> fuera a la izquierda
    return f"""    // color (recorte_color.py): cada transicion lee su tiempo del clip "Color" en el DOM
    const num = (el, k) => parseFloat(el.getAttribute(k) || "0");
    document.querySelectorAll("video[data-ancla]").forEach((v) => {{
      const w = v.parentElement, t0 = num(v, "data-start"), d = num(v, "data-trans-dur");
      const tipo = v.getAttribute("data-transicion");
      if (tipo === "cortinilla") {{
        tl.to(w, {{ clipPath: "{POLY_FIN}", duration: d, ease: "power2.inOut" }}, t0);
        const ui = document.getElementById(w.id + "-ui");
        if (ui) {{
          tl.to(ui, {{ opacity: 1, duration: 0.2 }}, Math.max(0, t0 - 0.2));
          tl.fromTo(ui.querySelector(".cw-linea"), {{ x: {x0:.1f}, rotation: {rot:.2f} }},
                    {{ x: {x1:.1f}, rotation: {rot:.2f}, duration: d, ease: "power2.inOut" }}, t0);
          tl.to(ui, {{ opacity: 0, duration: 0.3 }}, t0 + d + 0.35);
        }}
      }} else {{
        tl.to(w, {{ opacity: 1, duration: tipo === "fundido" ? d : 0.001 }}, t0);
      }}
    }});
"""


# ── from-studio ───────────────────────────────────────────────────────────────

def leer_sujetos(el: list, ranges: list, orig_idx: list, base_ranges: list, mapa: dict,
                 oculto: Callable[[dict], bool]) -> list[str]:
    """Pone ranges[k].sujeto segun Studio. El recorte se ubica por data-media-start (tiempo de la
    fuente), asi un ripple o un arrastre no lo desalinea; arrastrarlo se ignora (va pegado al clip)."""
    notas, emitidos, spans = [], mapa.get("sujetos") or {}, mapa.get("sujeto_assets") or {}
    fondos, ventanas = {}, {}
    for tag, a in el:
        if "fondo-sujeto" in (a.get("class") or "") and not oculto(a):
            m = re.search(r"background(?:-color)?:\s*([^;]+)", a.get("style", ""))
            fondos[int(a["data-sujeto"])] = m.group(1).strip() if m else None
    for tag, a in el:
        if tag != "video" or "data-sujeto" not in a or oculto(a):
            continue
        i, asset = int(a["data-sujeto"]), Path(a.get("src", "")).name
        if asset not in spans:
            notas.append(f"  {a['id']}: recorte con asset desconocido ({asset}) -> se ignora")
            continue
        st, ms, du = float(a.get("data-start", 0)), float(a.get("data-media-start", 0)), float(a["data-duration"])
        src0 = spans[asset]["span0"] + ms
        em = emitidos.get(a["id"])
        intacto = bool(em) and all(abs(x - y) <= 0.02 for x, y in ((em["start"], st), (em["dur"], du), (em["ms"], ms)))
        if em and abs(em["start"] - st) > 0.02 and abs(em["ms"] - ms) <= 0.02 and abs(em["dur"] - du) <= 0.02:
            notas.append(f"  {a['id']}: recorte arrastrado -> se ignora (va pegado a su clip)")
            intacto = True
        ventanas.setdefault(i, []).append((src0, src0 + du, intacto))
    for i in fondos:
        if i not in ventanas:
            notas.append(f"  fondo del sujeto del clip {i + 1} sin recorte -> se descarta (taparia a Jaime)")
    for k, r in enumerate(ranges):
        i = orig_idx[k]
        orig = base_ranges[i].get("sujeto")
        vs = ventanas.get(i)
        if not vs:
            if orig:
                notas.append(f"  clip {k + 1}: recorte de sujeto eliminado")
            continue
        a0, b0 = max(r["start"], min(v[0] for v in vs)), min(r["end"], max(v[1] for v in vs))
        if b0 - a0 < 0.05:
            if orig:
                notas.append(f"  clip {k + 1}: queda fuera de su recorte de sujeto (partido)")
            continue
        if all(v[2] for v in vs) and orig:
            s = dict(orig)
        else:
            s = {k: v for k, v in (orig or {"fondo": None}).items() if k != "src"}
            if a0 > r["start"] + 0.02 or b0 < r["end"] - 0.02:
                s["src"] = [round(a0, 3), round(b0, 3)]
            notas.append(f"  clip {k + 1}: recorte de sujeto {s.get('src', 'clip entero')}")
        fondo = fondos.get(i)
        if (fondo or None) != (s.get("fondo") or None):
            notas.append(f"  clip {k + 1}: fondo del sujeto {s.get('fondo')} -> {fondo}")
        s["fondo"] = fondo
        if len(vs) > 1:
            notas.append(f"  clip {k + 1}: recorte partido en {len(vs)} -> se usa la envolvente")
        r["sujeto"] = s
    return notas


def leer_color(el: list, wins: list, clip_attrs: list, mapa: dict, a_salida: Callable[[float], float],
               oculto: Callable[[dict], bool]) -> tuple[dict | None, list[str]]:
    """Reconstruye edl.color desde Studio: anclas (clips "Color") y el grade fijo de cada clip.
    -> (color nuevo o None si no cambio, notas)."""
    color0 = mapa.get("color")
    if not color0:
        return None, []
    notas = []
    looks = dict(color0.get("looks") or {})
    por_json = {look_json(v): k for k, v in looks.items()}

    def nombre(attr):
        if not attr:
            return None
        try:
            j = json.loads(attr)
        except ValueError:
            return None
        s = look_json(j)
        if s not in por_json:
            n = f"studio{sum(1 for k in looks if k.startswith('studio')) + 1}"
            looks[n], por_json[s] = j, n
            notas.append(f"  look nuevo editado en Studio -> '{n}'")
        return por_json[s]

    em = mapa.get("el_color") or {}
    anclas = []
    for tag, a in el:
        if tag != "video" or "data-ancla" not in a or oculto(a):
            continue
        st, info = float(a.get("data-start", 0)), em.get(a["id"])
        if info and abs(info["start"] - st) <= 0.02:
            src, source = float(a.get("data-media-start", 0)), info["source"]
        else:
            debajo = next(((s0, r) for s0, s1, _, _, r in wins if s0 - 1e-3 <= st < s1), None)
            if not debajo:
                notas.append(f"  {a['id']}: transicion de color fuera de los clips -> se descarta")
                continue
            src, source = debajo[1]["start"] + st - debajo[0], debajo[1]["source"]
            notas.append(f"  {a['id']}: transicion de color movida -> src {src:.2f}")
        anclas.append({"T": a_salida(st), "source": source, "src": round(src, 3),
                       "look": nombre(a.get("data-color-grading")),
                       "transicion": a.get("data-transicion", "corte"), "dur": float(a.get("data-trans-dur", 0))})
    anclas.sort(key=lambda x: x["T"])

    cambios, activo, j, fin_trans = [], color0.get("inicial"), 0, -1.0
    previo = activo
    for k, (s0, s1, o0, _, r) in enumerate(wins):
        o1 = o0 + (s1 - s0)
        visible = nombre(clip_attrs[k].get("data-color-grading"))
        esperado = previo if o0 < fin_trans - 0.02 else activo
        if visible != esperado:
            cambios.append({"source": r["source"], "src": round(r["start"], 3), "look": visible, "transicion": "corte"})
            activo = previo = visible
            fin_trans = -1.0
        while j < len(anclas) and anclas[j]["T"] < o1 - 1e-3:
            an = anclas[j]
            j += 1
            if an["look"] == activo:
                notas.append(f"  transicion a '{an['look']}' sin cambio de look -> se quita")
                continue
            c = {"source": an["source"], "src": an["src"], "look": an["look"], "transicion": an["transicion"]}
            if an["transicion"] != "corte":
                c["dur"] = round(an["dur"], 3)
            cambios.append(c)
            previo, activo, fin_trans = activo, an["look"], an["T"] + an["dur"]

    viejos = color0.get("cambios") or []
    igual = len(viejos) == len(cambios) and all(
        v.get("look") == c["look"] and v.get("transicion", "corte") == c["transicion"]
        and v["source"] == c["source"] and abs(v["src"] - c["src"]) <= 0.02 for v, c in zip(viejos, cambios))
    if igual and looks == (color0.get("looks") or {}):
        return None, notas
    # conserva los campos extra de los cambios que no se movieron (p. ej. "palabra")
    for c in cambios:
        for v in viejos:
            if v.get("look") == c["look"] and v["source"] == c["source"] and abs(v["src"] - c["src"]) <= 0.02:
                c.update({k: x for k, x in v.items() if k not in c})
                c["src"] = v["src"]
    notas.append(f"  color: {len(viejos)} -> {len(cambios)} cambios de look")
    return {**color0, "looks": looks, "cambios": cambios}, notas
