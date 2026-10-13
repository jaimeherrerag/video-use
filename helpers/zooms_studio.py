#!/usr/bin/env python3
"""zooms_studio.py - zooms animados (push-in / pull-out / punch) del Studio maestro (biblioteca de studio_maestro).

`ranges[i].zoom = {"de": 1.0, "a": 1.08, "src": [a, b]?, "origen": [x, y]?, "ease": "power1.inOut"?}`:
la camara del clip escala de `de` a `a` durante `src` (tiempo de la fuente; sin `src`, el clip entero),
con origen en `origen` (px del lienzo; por defecto la cara). Es propiedad del clip, asi que sigue sus
recortes y partidas. En Studio se ve animado y aparece como una marca "Zoom · 1.00->1.08" en el timeline:
moverla o recortarla cambia cuando pasa, borrarla lo quita (regla de oro, 2026-10-13).

Todo lo que se dibuja con la camara de un clip (sus copias de color y su recorte de sujeto) lleva
`data-zclip="<i>"` y escala igual, para que no se despeguen. Solo en clips sin `reframe` (el encuadre
fijo ya usa translate/clip-path; zoom y encuadre a la vez no se representan todavia).
"""

from __future__ import annotations

from html import escape
from typing import Callable

TRACK_ZOOM = 40
ORIGEN = (540, 760)                 # cara en el set actual (short 1080x1920)


def ventana_zoom(r: dict) -> tuple[float, float] | None:
    z = r.get("zoom")
    if not z:
        return None
    a, b = z.get("src") or [r["start"], r["end"]]
    a, b = max(a, r["start"]), min(b, r["end"])
    return (a, b) if b - a >= 0.05 else None


def html_zooms(ventanas: list, etq: Callable[[str], str]) -> tuple[list[str], dict]:
    """ventanas: [(t0, t1, r)] por clip. -> (marcas del timeline, mapa {id: emitido})."""
    lines, mapa, n = [], {}, 0
    for i, (t0, t1, r) in enumerate(ventanas):
        w = ventana_zoom(r)
        if not w:
            continue
        if r.get("reframe"):
            print(f"  aviso: el clip {i + 1} tiene encuadre fijo: su zoom no se emite")
            continue
        z = r["zoom"]
        a, b = w
        st, du = round(t0 + a - r["start"], 3), round(b - a, 3)
        de, hasta = float(z.get("de", 1.0)), float(z.get("a", 1.08))
        ox, oy = z.get("origen") or ORIGEN
        n += 1
        zid = f"z{n:03d}"
        lines.append(
            f'    <div id="{zid}" class="clip zoom-marca" data-zoom="{i}" data-de="{de:.3f}" data-a="{hasta:.3f}"'
            f' data-origen="{ox} {oy}" data-ease="{escape(z.get("ease", "power1.inOut"))}"\n'
            f'      data-timeline-label="{escape(etq(f"Zoom {de:.2f}→{hasta:.2f} (clip {i + 1})"))}"'
            f' data-start="{st:.3f}" data-duration="{du:.3f}" data-track-index="{TRACK_ZOOM + (n - 1) % 2}"></div>')
        mapa[zid] = {"range": i, "start": st, "dur": du, "de": de, "a": hasta}
    return lines, mapa


CSS = "    .zoom-marca { position: absolute; inset: 0; pointer-events: none; }\n"

JS = """    // zooms (zooms_studio.py): cada marca escala la camara de su clip (y sus copias de color y su recorte)
    document.querySelectorAll(".zoom-marca").forEach((m) => {
      const objetivos = document.querySelectorAll('[data-zclip="' + m.getAttribute("data-zoom") + '"]');
      if (!objetivos.length) return;
      const [ox, oy] = (m.getAttribute("data-origen") || "540 760").split(" ");
      tl.fromTo(objetivos, { scale: parseFloat(m.getAttribute("data-de")), transformOrigin: ox + "px " + oy + "px" },
                { scale: parseFloat(m.getAttribute("data-a")), duration: parseFloat(m.getAttribute("data-duration")),
                  ease: m.getAttribute("data-ease") || "power1.inOut" }, parseFloat(m.getAttribute("data-start")));
    });
"""


def leer_zooms(el: list, wins: list, ranges: list, orig_idx: list, base_ranges: list, mapa: dict,
               oculto: Callable[[dict], bool]) -> list[str]:
    """Pone ranges[k].zoom segun las marcas de Studio (tiempos -> fuente por el clip de debajo)."""
    notas, emitidos, vistos = [], mapa.get("zooms") or {}, set()
    for tag, a in el:
        if "zoom-marca" not in (a.get("class") or "") or oculto(a):
            continue
        i = int(a["data-zoom"])
        st, du = float(a.get("data-start", 0)), float(a["data-duration"])
        em = emitidos.get(a["id"])
        intacto = bool(em) and abs(em["start"] - st) <= 0.02 and abs(em["dur"] - du) <= 0.02
        de, hasta = float(a.get("data-de", 1)), float(a.get("data-a", 1))
        for k, (s0, s1, _, _, r) in enumerate(wins):
            if orig_idx[k] != i or s1 <= st or s0 >= st + du:
                continue
            z = {kk: v for kk, v in (base_ranges[i].get("zoom") or {}).items() if kk != "src"}
            z["de"], z["a"] = round(de, 3), round(hasta, 3)
            a0 = r["start"] + max(0.0, st - s0)
            b0 = min(r["end"], r["start"] + (st + du - s0))
            if intacto and base_ranges[i].get("zoom", {}).get("src") is not None:
                z["src"] = base_ranges[i]["zoom"]["src"]
            elif a0 > r["start"] + 0.02 or b0 < r["end"] - 0.02:
                z["src"] = [round(a0, 3), round(b0, 3)]
            if not intacto:
                notas.append(f"  {a['id']}: zoom del clip {k + 1} -> {z.get('src', 'clip entero')}")
            if em and (abs(em["de"] - de) > 1e-3 or abs(em["a"] - hasta) > 1e-3):
                notas.append(f"  {a['id']}: zoom {em['de']:.2f}->{em['a']:.2f} cambiado a {de:.2f}->{hasta:.2f}")
            ranges[k]["zoom"] = z
            vistos.add(k)
    for k, r in enumerate(ranges):
        if k not in vistos and base_ranges[orig_idx[k]].get("zoom"):
            notas.append(f"  clip {k + 1}: zoom eliminado")
    return notas
