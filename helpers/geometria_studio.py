"""geometria_studio.py - reframe del EDL <-> CSS de un <video> en HyperFrames Studio.

Regla de oro (CLAUDE.md): todo encuadre (recorte de pestanas, zoom) tiene que verse Y
editarse en Studio. Para eso el clip se escribe con el MISMO vocabulario que usa Studio al
escalarlo/moverlo con las manijas del canvas (medido 2026-10-08, 0.8.140):
  - escalar  -> cambia `width`/`height` del elemento + `translate` para conservar el centro,
               y escala `clip-path: inset(...)` en proporcion
  - mover    -> cambia `translate`
  (guarda los valores previos en atributos data-hf-studio-original-*; no crea keyframes)

Modelo: el lienzo es W x H (1920x1080 = el output). El <video> muestra la fuente ENTERA
(object-fit llena la caja: misma relacion de aspecto) en una caja de w_e x h_e px,
trasladada (tx, ty); el `clip-path` recorta en coordenadas de la caja. Un pixel u de la
fuente cae en el lienzo en  tx + u * w_e / SW.

Reframe "place" (el general; crop-pad y crop-scale son casos particulares):
    {"src_crop": {x,y,w,h} en px de la fuente, "dst": {x,y,w,h} en px del output,
     "out_size": {w,h}, "fit": "place"}
"""

from __future__ import annotations

import re

W, H = 1920, 1080


def place_de_reframe(rf: dict | None, sw: int, sh: int, w: int = W, h: int = H) -> dict | None:
    """Normaliza cualquier reframe del EDL a (src_crop, dst) en px. None = sin reframe."""
    if not rf:
        return None
    c = rf["src_crop"]
    cx, cy, cw, ch = float(c["x"]), float(c["y"]), float(c["w"]), float(c["h"])
    fit = rf.get("fit", "blur-bg")
    if fit == "place":
        d = rf["dst"]
        return {"src": (cx, cy, cw, ch), "dst": (float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"]))}
    ow = float(rf.get("out_size", {}).get("w", w))
    oh = float(rf.get("out_size", {}).get("h", h))
    if fit == "crop-scale":
        return {"src": (cx, cy, cw, ch), "dst": (0.0, 0.0, w, h)}
    if fit == "crop-pad":
        s = min(1.0, ow / cw, oh / ch)          # render.py: reduce solo si no cabe
        dw, dh = cw * s * w / ow, ch * s * h / oh
        return {"src": (cx, cy, cw, ch), "dst": ((w - dw) / 2, (h - dh) / 2, dw, dh)}
    return None                                   # blur-bg (shorts): sin equivalente aqui


def css_de_place(p: dict | None, sw: int, sh: int) -> str:
    """CSS inline del <video> para un place (o identidad si p es None)."""
    if p is None:
        return ""
    cx, cy, cw, ch = p["src"]
    dx, dy, dw, dh = p["dst"]
    m = dw / cw                                   # px de lienzo por px de fuente (uniforme)
    we, he = sw * m, sh * m
    tx, ty = dx - cx * m, dy - cy * m
    top, left = cy * m, cx * m
    right, bottom = we - (cx + cw) * m, he - (cy + ch) * m
    return (f"width: {we:.1f}px; height: {he:.1f}px; translate: {tx:.1f}px {ty:.1f}px; "
            f"clip-path: inset({top:.1f}px {right:.1f}px {bottom:.1f}px {left:.1f}px);")


def _px(v: str) -> float:
    return float(v.strip().replace("px", "") or 0)


def _origen(v: str, size: float) -> float:
    v = v.strip()
    if v.endswith("%"):
        return float(v[:-1]) / 100 * size
    if v in ("left", "top"):
        return 0.0
    if v in ("right", "bottom"):
        return size
    if v == "center":
        return size / 2
    return _px(v)


def parse_style(style: str) -> dict:
    out = {}
    for decl in (style or "").split(";"):
        if ":" in decl:
            k, v = decl.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def geometria(style: str, sw: int, sh: int, w: int = W, h: int = H):
    """Lee el CSS que dejo Studio y devuelve una funcion caja->lienzo y la caja visible.

    Soporta lo que escribimos (width/height/translate/clip-path) y lo que agrega Studio
    (lo mismo + transform translate()/scale() con transform-origin, que dejaba el formato
    anterior de studio_cuts).
    """
    st = parse_style(style)
    we = _px(st["width"]) if "width" in st and st["width"].endswith("px") else float(w)
    he = _px(st["height"]) if "height" in st and st["height"].endswith("px") else float(h)
    tx = ty = 0.0
    if "translate" in st:
        parts = st["translate"].split()
        tx = _px(parts[0])
        ty = _px(parts[1]) if len(parts) > 1 else 0.0
    ax = ay = 0.0
    sx = sy = 1.0
    ox = oy = 0.0
    if "transform" in st:
        m = re.search(r"translate\(\s*([-\d.]+)px\s*,\s*([-\d.]+)px\s*\)", st["transform"])
        if m:
            ax, ay = float(m.group(1)), float(m.group(2))
        m = re.search(r"scale\(\s*([-\d.]+)\s*(?:,\s*([-\d.]+))?\s*\)", st["transform"])
        if m:
            sx = float(m.group(1))
            sy = float(m.group(2)) if m.group(2) else sx
        o = st.get("transform-origin", "50% 50%").split()
        ox = _origen(o[0] if o else "50%", we)
        oy = _origen(o[1] if len(o) > 1 else "50%", he)
    t = r = b = l = 0.0
    if "clip-path" in st:
        m = re.search(r"inset\(([^)]*)\)", st["clip-path"])
        if m:
            v = [_px(x) for x in m.group(1).split()]
            # shorthand CSS: 1, 2, 3 o 4 valores (top right bottom left)
            t, r, b, l = {1: (v[0], v[0], v[0], v[0]), 2: (v[0], v[1], v[0], v[1]),
                          3: (v[0], v[1], v[2], v[1]), 4: tuple(v[:4])}[min(len(v), 4)]

    def a_lienzo(px: float, py: float) -> tuple[float, float]:
        return (tx + ax + ox + sx * (px - ox), ty + ay + oy + sy * (py - oy))

    caja = (l, t, we - r, he - b)               # rect visible en coords de la caja
    return a_lienzo, caja, (we, he), (sx, sy)


def place_de_css(style: str, sw: int, sh: int, w: int = W, h: int = H) -> dict | None:
    """Inverso de css_de_place: que parte de la fuente se ve y en que rect del output."""
    a_lienzo, (bl, bt, br, bb), (we, he), (sx, sy) = geometria(style, sw, sh, w, h)
    x0, y0 = a_lienzo(bl, bt)
    x1, y1 = a_lienzo(br, bb)
    # interseccion con el lienzo (root overflow:hidden)
    dx0, dy0, dx1, dy1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return {"oculto": True}
    # de lienzo a caja a fuente
    kx, ky = sw / (we * sx), sh / (he * sy)
    sx0 = bl * (sw / we) + (dx0 - x0) * kx
    sy0 = bt * (sh / he) + (dy0 - y0) * ky
    sw_ = (dx1 - dx0) * kx
    sh_ = (dy1 - dy0) * ky
    p = {"src": (sx0, sy0, sw_, sh_), "dst": (dx0, dy0, dx1 - dx0, dy1 - dy0)}
    ident = (abs(sx0) < 2 and abs(sy0) < 2 and abs(sw_ - sw) < 4 and abs(sh_ - sh) < 4
             and abs(dx0) < 1 and abs(dy0) < 1 and abs(dx1 - w) < 1 and abs(dy1 - h) < 1)
    return None if ident else p


def reframe_de_place(p: dict, w: int = W, h: int = H) -> dict:
    """place -> reframe del EDL (pixeles enteros y pares, como pide ffmpeg)."""
    def par(v: float) -> int:
        return max(2, int(round(v / 2)) * 2)
    cx, cy, cw, ch = p["src"]
    dx, dy, dw, dh = p["dst"]
    return {"src_crop": {"x": int(round(cx)), "y": int(round(cy)), "w": par(cw), "h": par(ch)},
            "dst": {"x": int(round(dx)), "y": int(round(dy)), "w": par(dw), "h": par(dh)},
            "out_size": {"w": w, "h": h}, "fit": "place"}


def rect_div(style: str) -> tuple[float, float, float, float]:
    """Rect en el lienzo de un <div> posicionado (left/top/width/height + translate)."""
    st = parse_style(style)
    x, y = _px(st.get("left", "0")), _px(st.get("top", "0"))
    wd, ht = _px(st.get("width", "0")), _px(st.get("height", "0"))
    if "translate" in st:
        parts = st["translate"].split()
        x += _px(parts[0])
        y += _px(parts[1]) if len(parts) > 1 else 0.0
    return x, y, wd, ht


def lienzo_a_fuente(place: dict | None, rect, sw: int, sh: int, w: int = W, h: int = H):
    """Rect del lienzo -> rect en px de la fuente, segun el encuadre del clip."""
    x, y, wd, ht = rect
    if place is None:
        k = sw / w
        return x * k, y * (sh / h), wd * k, ht * (sh / h)
    cx, cy, cw, ch = place["src"]
    dx, dy, dw, dh = place["dst"]
    k = cw / dw
    return cx + (x - dx) * k, cy + (y - dy) * k, wd * k, ht * k


def fuente_a_lienzo(place: dict | None, rect, sw: int, sh: int, w: int = W, h: int = H):
    x, y, wd, ht = rect
    if place is None:
        k = w / sw
        return x * k, y * (h / sh), wd * k, ht * (h / sh)
    cx, cy, cw, ch = place["src"]
    dx, dy, dw, dh = place["dst"]
    k = dw / cw
    return dx + (x - cx) * k, dy + (y - cy) * k, wd * k, ht * k
