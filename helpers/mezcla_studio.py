#!/usr/bin/env python3
"""mezcla_studio.py - revisar la mezcla (musica + SFX) en HyperFrames Studio, ida y vuelta
con el stitch.json de stitch_final.py.

La musica y los SFX se mezclan recien en stitch_final.py, asi que Studio (la intro, los
cortes) nunca los muestra. Este helper arma un proyecto de Studio SOLO para la mezcla:
el video completo ANTES del speed (proxy 540p, intro con tarjetas + resto), la voz como
referencia a volumen 1.0, y cada pista de musica / cada SFX como un <audio> con su
volumen relativo actual (el fader de Studio = `data-volume`, 0..3.98).

    to-studio   --spec <edit>/stitch/stitch.json   -> <edit>/mezcla/ (+ mezcla_map.json)
    from-studio --spec <edit>/stitch/stitch.json   -> reescribe stitch.json (respaldo .bak)

Conversion (G = ganancia del loudnorm medida sobre la voz, la misma que usa el stitch):
    musica  data-volume = 10^((target_db - G - media_pista)/20)
    SFX     data-volume = 10^((peak_db  - G - pico_archivo)/20)
y de vuelta, relativo al volumen que Jaime deje en la voz. Los tiempos van en el timeline
ANTES del speed (el de stitch.json). Los fades de la musica se simulan con una etapa
`gain` automatizada (se multiplica con el fader; solo para escuchar: el stitch usa los
fade_in/fade_out del JSON). Studio reproduce a 1x: el entregable va a `speed`.

El proxy (assets/mezcla_proxy.mp4) se genera aparte: concat de las partes del stitch a
960x540 con la voz en AAC (ver CLAUDE.md, "Mezcla en Studio").
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from html import escape
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stitch_final import cargar_partes, dur, medir_ganancia, nivel, pico_s  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PRESET = REPO / "presets" / "glass-composition"   # package.json / hyperframes.json pineados


def fmt(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def clips_audio(spec: dict, base: Path, assets: Path, g: float, speed: float,
                track0: int = 2) -> tuple[list[str], dict]:
    """<audio> de musica y SFX (volumen relativo a la voz = 1) + el mapa para leerlos de vuelta.
    Lo usan este CLI y studio_maestro.py (regla de oro: la mezcla se ve y se edita en Studio)."""
    clips, mapa = [], {"ganancia": g, "speed": speed, "music": [], "sfx": []}
    for i, m in enumerate(spec.get("music", [])):
        f = (base / m["file"]).resolve()
        dst = assets / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
        mean, _ = nivel(f)
        vol = 10 ** ((m["target_db"] - g - mean) / 20)
        # Studio va a 1x: la pista puede no alcanzar para todo el tramo antes del speed
        d = round(min(m["dur"], dur(f, "a:0") - m.get("src", 0)), 3)
        lane = [{"t": 0, "v": -60 if m.get("fade_in") else 0}]
        if m.get("fade_in"):
            lane.append({"t": round(m["fade_in"], 3), "v": 0})
        if m.get("fade_out"):
            lane += [{"t": round(d - m["fade_out"], 3), "v": 0}, {"t": round(d, 3), "v": -60}]
        fx = json.dumps({"version": 1, "nodes": [{"type": "gain", "id": "fade", "params": {"gain": 0}}]})
        auto = json.dumps({"version": 1, "lanes": [{"target": "fx.fade.gain", "points": lane}]})
        cid = f"musica-{i}"
        clips.append(
            f'    <audio id="{cid}" src="assets/{escape(f.name)}" data-audio-group="musica"\n'
            f'      data-timeline-label="Música {i + 1} · {escape(f.stem[:30])} ({fmt(m["t"])})"\n'
            f'      data-start="{m["t"]:.3f}" data-duration="{d:.3f}" data-media-start="{m.get("src", 0):.3f}"\n'
            f'      data-volume="{vol:.4f}" data-track-index="{track0 + i}"\n'
            f"      data-fx-chain='{fx}'\n      data-automation='{auto}'></audio>")
        mapa["music"].append({"id": cid, "mean": mean, "ini": {"start": round(m["t"], 3), "dur": d,
                              "ms": round(m.get("src", 0), 3), "vol": round(vol, 4)}})
    for i, s in enumerate(spec.get("sfx", [])):
        f = (base / s["file"]).resolve()
        dst = assets / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
        _, peak = nivel(f)
        vol = 10 ** ((s["peak_db"] - g - peak) / 20)
        # mismo lugar que el stitch: alinear se mide en segundos REALES del archivo, que en el
        # timeline antes del speed equivalen a offset * speed
        off = {"pico": pico_s(f), "fin": dur(f, "a:0")}.get(s.get("alinear"), 0.0)
        start = max(0.0, s["t"] - off * speed)
        cid = f"sfx-{i:02d}"
        clips.append(
            f'    <audio id="{cid}" src="assets/{escape(f.name)}" data-audio-group="sfx"\n'
            f'      data-timeline-label="SFX {i + 1} · {escape(f.stem)} ({fmt(s["t"])})"\n'
            f'      data-start="{start:.3f}" data-duration="{dur(f, "a:0"):.3f}"\n'
            f'      data-volume="{min(vol, 3.98):.4f}" data-track-index="{track0 + 2 + i % 2}"></audio>')
        mapa["sfx"].append({"id": cid, "peak": peak, "off": off,
                            "ini": {"start": round(start, 3), "vol": round(min(vol, 3.98), 4)}})

    return clips, mapa


def to_studio(spec_path: Path) -> None:
    base = spec_path.parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    speed = float(spec.get("speed", 1.0))
    ln = spec.get("loudnorm", {"I": -14, "TP": -1, "LRA": 11})
    edit = base.parent
    out = edit / "mezcla"
    assets = out / "assets"
    proxy = assets / "mezcla_proxy.mp4"
    if not proxy.exists():
        sys.exit(f"falta el proxy {proxy} (ver docstring)")

    parts = cargar_partes(spec, base)
    total = sum(d for _, _, d in parts)
    g = medir_ganancia(parts, speed, ln)

    clips, mapa = clips_audio(spec, base, assets, g, speed)

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=1920, height=1080"/>
  <title>Mezcla</title>
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
  <style>
    html, body {{ margin: 0; width: 1920px; height: 1080px; overflow: hidden; background: #000; }}
    #root {{ position: relative; width: 1920px; height: 1080px; overflow: hidden; }}
    #proxy {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <!-- Generado por video-use/helpers/mezcla_studio.py: SOLO para revisar la mezcla.
       Jaime mueve faders (data-volume) y tiempos; from-studio los lleva a stitch.json.
       Video a 1x (antes del speed {speed:g}); la voz es la referencia (volumen 1). -->
  <div id="root" data-composition-id="mezcla" data-start="0" data-duration="{total:.3f}"
    data-width="1920" data-height="1080">
    <video id="proxy" src="assets/mezcla_proxy.mp4" data-start="0" data-duration="{total:.3f}"
      data-track-index="0" muted playsinline></video>
    <audio id="voz" src="assets/mezcla_proxy.mp4" data-timeline-label="Voz (referencia)"
      data-start="0" data-duration="{total:.3f}" data-volume="1" data-track-index="1"></audio>
{chr(10).join(clips)}
  </div>
  <script>
    const tl = gsap.timeline({{ paused: true }});
    tl.to({{}}, {{ duration: {total:.3f} }}, 0);
    window.__timelines["mezcla"] = tl;
  </script>
</body>
</html>
"""
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "mezcla_map.json").write_text(json.dumps(mapa, indent=2), encoding="utf-8")
    pkg = json.loads((PRESET / "package.json").read_text(encoding="utf-8"))
    pkg["name"] = "mezcla"
    (out / "package.json").write_text(json.dumps(pkg, indent=2), encoding="utf-8")
    shutil.copy(PRESET / "hyperframes.json", out / "hyperframes.json")
    print(f"mezcla: voz + {len(mapa['music'])} musica + {len(mapa['sfx'])} SFX, {total:.1f}s -> {out / 'index.html'}")


class Audios(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.a: dict[str, dict] = {}

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "audio" and d.get("id"):
            self.a[d["id"]] = d


def leer_audio(spec: dict, mapa: dict, audios: dict, vref: float = 1.0) -> tuple[list, list, list[str]]:
    """De los <audio> que dejo Studio a music/sfx del stitch.json. Solo aplica lo que se
    movio respecto del valor inicial (mapa[...]["ini"]); devuelve tambien el reporte."""
    g, speed = mapa["ganancia"], mapa["speed"]

    def oculto(a: dict) -> bool:
        return "data-hidden" in a and a.get("data-hidden") not in ("false", "0")

    def db(a: dict) -> float:
        v = float(a.get("data-volume", 1)) / vref
        return 20 * math.log10(max(v, 1e-6))

    cambios, music, sfx = [], [], []
    for m, info in zip(spec.get("music", []), mapa["music"]):
        a = audios.get(info["id"])
        if a is None or oculto(a):
            cambios.append(f"  {info['id']}: ELIMINADA")
            continue
        nuevo, ini = dict(m), info["ini"]
        cur = lambda k, d: float(a.get(k, d))
        if abs(cur("data-volume", ini["vol"]) - ini["vol"]) > 1e-4 or abs(vref - 1) > 1e-3:
            nuevo["target_db"] = round(db(a) + g + info["mean"], 1)
        if abs(cur("data-start", ini["start"]) - ini["start"]) > 0.01:
            nuevo["t"] = round(cur("data-start", 0), 3)
        if abs(cur("data-duration", ini["dur"]) - ini["dur"]) > 0.01:
            nuevo["dur"] = round(cur("data-duration", 0), 3)
        if abs(cur("data-media-start", ini["ms"]) - ini["ms"]) > 0.01:
            nuevo["src"] = round(cur("data-media-start", 0), 3)
        for k in ("target_db", "t", "dur", "src"):
            if abs(nuevo.get(k, 0) - m.get(k, 0)) > 0.05:
                cambios.append(f"  {info['id']}: {k} {m.get(k)} -> {nuevo[k]}")
        music.append(nuevo)
    for s, info in zip(spec.get("sfx", []), mapa["sfx"]):
        a = audios.get(info["id"])
        if a is None or oculto(a):
            cambios.append(f"  {info['id']} ({Path(s['file']).stem}): ELIMINADO")
            continue
        nuevo, ini = dict(s), info["ini"]
        cur = lambda k, d: float(a.get(k, d))
        if abs(cur("data-volume", ini["vol"]) - ini["vol"]) > 1e-4 or abs(vref - 1) > 1e-3:
            nuevo["peak_db"] = round(db(a) + g + info["peak"], 1)
        if abs(cur("data-start", ini["start"]) - ini["start"]) > 0.01:
            nuevo["t"] = round(cur("data-start", 0) + info["off"] * speed, 3)
        for k in ("peak_db", "t"):
            if abs(nuevo[k] - s[k]) > 0.05:
                cambios.append(f"  {info['id']} ({Path(s['file']).stem}): {k} {s[k]} -> {nuevo[k]}")
        sfx.append(nuevo)
    if abs(vref - 1) > 1e-3:
        cambios.insert(0, f"  voz a {vref:.2f} en Studio: musica y SFX se leen RELATIVOS a la voz")
    return music, sfx, cambios


def from_studio(spec_path: Path) -> None:
    base = spec_path.parent
    out = base.parent / "mezcla"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    mapa = json.loads((out / "mezcla_map.json").read_text(encoding="utf-8"))
    p = Audios()
    p.feed((out / "index.html").read_text(encoding="utf-8"))
    vref = float(p.a.get("voz", {}).get("data-volume", 1) or 1)

    music, sfx, cambios = leer_audio(spec, mapa, p.a, vref)

    print("\n".join(cambios) if cambios else "  sin cambios en la mezcla")
    if cambios:
        shutil.copy2(spec_path, spec_path.with_suffix(".json.bak"))
        spec["music"], spec["sfx"] = music, sfx
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  escrito {spec_path} (respaldo .json.bak)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["to-studio", "from-studio"])
    ap.add_argument("--spec", required=True, type=Path)
    args = ap.parse_args()
    (to_studio if args.cmd == "to-studio" else from_studio)(args.spec.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
