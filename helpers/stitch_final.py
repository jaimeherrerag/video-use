#!/usr/bin/env python3
"""stitch_final.py - une tramos (HyperFrames + render.py) en el entregable final.

Reemplaza al `stitch_final.ps1` que se quedo en la PC vieja. Implementa las reglas del
CLAUDE.md ("Liquid-glass solo en un tramo -> render parcial + stitch" y "Musica y
efectos de sonido"):

  - VIDEO: de cada parte (render de HF o salida --pcm de render.py), recortado a la
    duracion de su base PCM (HF puede dejar cola negra); concat FILTER con re-encode y
    fps/dimensiones normalizados (nunca concat -c copy entre encoders distintos).
  - AUDIO: SIEMPRE del base PCM de cada parte (nunca el AAC que mete HF), recortado a la
    misma duracion que su video. Una sola generacion AAC, al final, a 256k.
  - SPEED: setpts/atempo sobre todo el entregable (en el stitch, no en render.py).
  - MUSICA y SFX: se mezclan DESPUES del speed (la musica no se acelera); sus tiempos se
    dan en el timeline de salida ANTES del speed y aqui se dividen entre `speed`.
    Volumen de cada SFX calculado desde su pico medido: volume = 10^((objetivo-pico)/20).
  - LOUDNORM: 2 pasadas, la segunda LINEAL (el 1-pass bombea la musica en pausas).

Uso:
    uv run python helpers/stitch_final.py <edit>/stitch/stitch.json [--solo-audio]

Spec (rutas relativas a la carpeta del JSON):
{
  "fps": 30, "speed": 1.08, "out": "../final.mp4",
  "parts": [ {"video": "...mp4", "audio": "...pcm.mp4"}, ... ],
  "music": [ {"file": "m.mp3", "t": 0, "dur": 84.8, "src": 0, "fade_in": 0, "fade_out": 2,
              "target_db": -41} ],            # media EFECTIVA en el entregable (ya con loudnorm)
  "sfx":   [ {"file": "s.wav", "t": 43.2, "peak_db": -7.5} ],   # pico EFECTIVO (ya con loudnorm)
  "loudnorm": {"I": -14, "TP": -1, "LRA": 11}
}
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        sys.exit(f"ffmpeg fallo:\n{' '.join(cmd)}\n{p.stderr[-3000:]}")
    return p.stderr


def dur(path: Path, stream: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
                          "stream=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return float(out[0])


def nivel(path: Path) -> tuple[float, float]:
    """(mean_volume, max_volume) en dB."""
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "volumedetect",
               "-f", "null", "-"])
    mean = float(re.search(r"mean_volume:\s*(-?[\d.]+)", err).group(1))
    peak = float(re.search(r"max_volume:\s*(-?[\d.]+)", err).group(1))
    return mean, peak


def pico_s(path: Path) -> float:
    """Segundo del archivo donde esta su ventana (50 ms) mas fuerte: el 'golpe' del SFX."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000",
                          "-f", "s16le", "-"], capture_output=True).stdout
    a = np.abs(np.frombuffer(raw, np.int16).astype(np.float32))
    w = 400
    n = len(a) // w
    rms = (a[: n * w].reshape(n, w) ** 2).mean(axis=1)
    return float(rms.argmax() * w / 8000)


def cargar_partes(spec: dict, base: Path, verbose: bool = False) -> list[tuple[Path, Path, float]]:
    """(video, audio base, duracion). La duracion es la del VIDEO del base PCM."""
    parts = []
    for p in spec["parts"]:
        v, a = (base / p["video"]).resolve(), (base / p["audio"]).resolve()
        d, dv = dur(a, "v:0"), dur(v, "v:0")
        if dv + 0.02 < d:
            sys.exit(f"{v.name} dura {dv:.3f}s y su base {d:.3f}s: el video quedaria corto")
        parts.append((v, a, d))
        if verbose:
            print(f"  parte {v.name}: {d:.3f}s (video fuente {dv:.3f}s)")
    return parts


def cadena_voz(parts, speed: float) -> tuple[list[str], list[str], int]:
    """Inputs y filtergraph que dejan la voz concatenada y acelerada en [voz]."""
    inputs, fc = [], []
    for k, (_, a, d) in enumerate(parts):
        inputs += ["-i", str(a)]
        fc.append(f"[{k}:a:0]atrim=0:{d:.3f},asetpts=PTS-STARTPTS,aresample=48000[p{k}]")
    n = len(parts)
    fc.append("".join(f"[p{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1"
              + (f",atempo={speed:g}" if abs(speed - 1) > 1e-6 else "") + "[voz]")
    return inputs, fc, n


def medir_ganancia(parts, speed: float, ln: dict) -> float:
    """dB que el loudnorm final le sumara a ESTA voz (se mide, no se asume).

    En 2026-08 fue +2.5 dB y en el video del 2026-10-07 +4.99 dB: con un valor fijo la
    musica quedo 2 dB arriba del objetivo. La musica y los SFX casi no mueven la
    integrada, asi que basta medir la voz sola. mezcla_studio.py usa la misma funcion.
    """
    inputs, fc, _ = cadena_voz(parts, speed)
    err = run(["ffmpeg", "-hide_banner", "-nostats", *inputs, "-filter_complex",
               ";".join(fc) + f";[voz]loudnorm=I={ln['I']}:TP={ln['TP']}:LRA={ln['LRA']}:"
               "print_format=json[o]", "-map", "[o]", "-f", "null", "-"])
    mv = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
    g = ln["I"] - float(mv["input_i"])
    print(f"  voz sola: {mv['input_i']} LUFS -> el loudnorm la sube {g:+.2f} dB")
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description="Stitch final: partes + speed + musica + SFX + loudnorm")
    ap.add_argument("spec", type=Path)
    ap.add_argument("--solo-audio", action="store_true", help="solo genera mix.wav (para revisar la mezcla)")
    args = ap.parse_args()

    spec_path = args.spec.resolve()
    base = spec_path.parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    fps, speed = int(spec.get("fps", 30)), float(spec.get("speed", 1.0))
    out = (base / spec["out"]).resolve()
    ln = spec.get("loudnorm", {"I": -14, "TP": -1, "LRA": 11})

    # --- 1. partes: duracion = la del VIDEO del base PCM (la cola de HF se descarta)
    parts = cargar_partes(spec, base, verbose=True)
    total = sum(d for _, _, d in parts)
    print(f"  total {total:.3f}s -> con speed {speed:g}: {total / speed:.3f}s")

    # --- 2. mezcla de audio (PCM): voz acelerada + musica + SFX ------------------------
    mix = out.with_name(out.stem + "_mix.wav")
    inputs, fc, k = cadena_voz(parts, speed)
    capas = ["[voz]"]
    ganancia = 0.0
    if spec.get("music") or spec.get("sfx"):
        ganancia = medir_ganancia(parts, speed, ln)

    for m in spec.get("music", []):
        f = (base / m["file"]).resolve()
        mean, _ = nivel(f)
        # target_db = media EFECTIVA de la musica en el entregable (despues del loudnorm)
        vol = 10 ** ((m["target_db"] - ganancia - mean) / 20)
        t, d = m["t"] / speed, m["dur"] / speed
        fade_out_st = max(0.0, d - m.get("fade_out", 0))
        inputs += ["-i", str(f)]
        chain = (f"[{k}:a]atrim={m.get('src', 0):.3f}:{m.get('src', 0) + d:.3f},asetpts=PTS-STARTPTS,"
                 f"aresample=48000,volume={vol:.5f}")
        if m.get("fade_in"):
            chain += f",afade=t=in:st=0:d={m['fade_in']}"
        if m.get("fade_out"):
            chain += f",afade=t=out:st={fade_out_st:.3f}:d={m['fade_out']}"
        ms = int(round(t * 1000))
        fc.append(chain + f",adelay={ms}|{ms}[m{k}]")
        capas.append(f"[m{k}]")
        print(f"  musica {f.name}: {t:.2f}-{t + d:.2f}s (final) media {mean:.1f} dB -> volume {vol:.4f}")
        k += 1

    for s in spec.get("sfx", []):
        f = (base / s["file"]).resolve()
        _, peak = nivel(f)
        # peak_db = pico EFECTIVO en el entregable. La calibracion del CLAUDE.md (2026-09-03:
        # whoosh -8 / subdrop -10 / HUD -12) asumia un loudnorm de +2.5 dB, o sea -5.5/-7.5/-9.5
        # efectivos. Con la ganancia medida se conserva el balance contra la voz en cualquier video.
        vol = 10 ** ((s["peak_db"] - ganancia - peak) / 20)
        # "alinear": "pico" -> el golpe del archivo cae en t (whooshes en el corte);
        # "fin" -> el archivo TERMINA en t (riser que desemboca en un reveal)
        t0 = s["t"] / speed
        if s.get("alinear") == "pico":
            t0 -= pico_s(f)
        elif s.get("alinear") == "fin":
            t0 -= dur(f, "a:0")
        ms = int(round(max(0.0, t0) * 1000))
        inputs += ["-i", str(f)]
        fc.append(f"[{k}:a]aresample=48000,volume={vol:.4f},adelay={ms}|{ms}[s{k}]")
        capas.append(f"[s{k}]")
        print(f"  sfx {f.name}: t={s['t'] / speed:.2f}s pico {peak:.1f} -> {s['peak_db']} dB (volume {vol:.3f})")
        k += 1

    fc.append(f"{''.join(capas)}amix=inputs={len(capas)}:duration=first:normalize=0[mix]")
    run(["ffmpeg", "-y", "-hide_banner", *inputs, "-filter_complex", ";".join(fc),
         "-map", "[mix]", "-c:a", "pcm_s16le", "-ar", "48000", str(mix)])
    print(f"  mezcla -> {mix.name} ({dur(mix, 'a:0'):.3f}s)")
    if args.solo_audio:
        return 0

    # --- 3. loudnorm pasada 1 (medicion) sobre la mezcla ---------------------------------
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(mix), "-af",
               f"loudnorm=I={ln['I']}:TP={ln['TP']}:LRA={ln['LRA']}:print_format=json", "-f", "null", "-"])
    m = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
    lin = (f"loudnorm=I={ln['I']}:TP={ln['TP']}:LRA={ln['LRA']}:measured_I={m['input_i']}:"
           f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
           f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    print(f"  loudnorm medido: I={m['input_i']} TP={m['input_tp']} LRA={m['input_lra']}")

    # --- 4. encode final: video concat + speed, audio = mezcla normalizada ---------------
    inputs, fc = [], []
    for i, (v, _, d) in enumerate(parts):
        inputs += ["-i", str(v)]
        fc.append(f"[{i}:v:0]trim=0:{d:.3f},setpts=PTS-STARTPTS,fps={fps},scale=1920:1080:flags=lanczos,"
                  f"setsar=1,format=yuv420p[v{i}]")
    vs = "".join(f"[v{i}]" for i in range(len(parts)))
    fc.append(f"{vs}concat=n={len(parts)}:v=1:a=0"
              + (f",setpts=PTS/{speed:g},fps={fps}" if abs(speed - 1) > 1e-6 else "") + "[vout]")
    inputs += ["-i", str(mix)]
    fc.append(f"[{len(parts)}:a]{lin},aresample=48000[aout]")
    run(["ffmpeg", "-y", "-hide_banner", *inputs, "-filter_complex", ";".join(fc),
         "-map", "[vout]", "-map", "[aout]",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(fps),
         "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", "-shortest", str(out)])
    dv, da = dur(out, "v:0"), dur(out, "a:0")
    print(f"listo: {out} | video {dv:.3f}s audio {da:.3f}s (diferencia {abs(dv - da) * 1000:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
