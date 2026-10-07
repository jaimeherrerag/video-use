"""retx_windows.py - re-transcribe las ventanas con voz SIN words y las integra.

Fix (b) de la regresion 2026-08-04 del CLAUDE.md: el fallo principal de WhisperX
es omitir habla (la tapa estirando una word vecina, o se salta una repeticion o un
"borra eso" corto). Este script:

  1. busca en TODO el source los tramos de voz (mapa de energia de verify_edl, -38 dB)
     que ninguna word explica (cobertura clampeada) y que suenan a habla (pico > -30 dB)
  2. los agrupa en ventanas con ~4 s de contexto y boundaries EN SILENCIO
     (si el clip arranca a mitad de frase, whisper comprime los timestamps)
  3. re-transcribe cada ventana con WhisperX (modelo cargado una sola vez, sobre el
     WAV en memoria) y mete TODAS sus words al timeline en lugar de las viejas,
     salvo que la re-transcripcion traiga menos words (entonces se queda la vieja)

Corre en .venv-asr:
    .venv-asr\\Scripts\\python.exe helpers/retx_windows.py --edit-dir <edit> \\
        --source <video> --audio <edit>/mic16k.wav [--key raw] [--dry-run]

Respaldo: la primera vez copia transcripts/<key>.json -> <key>.orig.json y siempre
parte de ese original (idempotente). Reporte: <edit>/retx_report.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
sys.path.insert(0, str(Path(__file__).parent))
import verify_edl as V  # noqa: E402
from transcribe_local import DEFAULT_PROMPT, INTERP_LOGPROB, interpolate_missing, to_scribe_schema  # noqa: E402

STEP = V.VENTANA_MS / 1000.0
SR = 16000
CONTEXTO = 4.0       # s de contexto a cada lado del tramo sospechoso
AGRUPAR = 4.0        # tramos a menos de esto van en la misma ventana
QUIETO_MIN = 0.40    # silencio minimo para poner un boundary
BUSCAR_QUIETO = 10.0  # cuanto se aleja como maximo buscando ese silencio


def punto_quieto(db, t: float, hacia: int) -> float:
    """Centro del primer silencio >= QUIETO_MIN desde t hacia atras (-1) o adelante (+1)."""
    n_min = int(QUIETO_MIN / STEP)
    i = int(t / STEP)
    lim = int(BUSCAR_QUIETO / STEP)
    racha = 0
    for k in range(lim):
        j = i + hacia * k
        if j < 0 or j >= len(db):
            return 0.0 if j < 0 else len(db) * STEP
        racha = racha + 1 if db[j] <= V.VOZ_DB else 0
        if racha >= n_min:
            centro = j - hacia * (racha // 2)
            return centro * STEP
    return t


def candidatos(words, db, min_voz: float):
    """Tramos de voz sin ninguna word que los explique."""
    cubierto = sorted(V.cobertura(w, db) for w in words)
    voz = [r for r in V.runs_voz(db, 0.0, len(db) * STEP) if r[1] - r[0] >= V.BLIP_MAX]
    voz = V.fusionar_runs(voz)
    out, k = [], 0
    for rs, re_ in voz:
        while k < len(cubierto) and cubierto[k][1] < rs - 2.0:
            k += 1
        libre = [(rs, re_)]
        for cs, ce in cubierto[k:]:
            if cs > re_:
                break
            nuevo = []
            for ls, le in libre:
                if ce <= ls or cs >= le:
                    nuevo.append((ls, le))
                    continue
                if cs > ls:
                    nuevo.append((ls, cs))
                if ce < le:
                    nuevo.append((ce, le))
            libre = nuevo
        for ls, le in libre:
            if le - ls >= min_voz and V.db_pico(db, ls, le) > V.HABLA_DB:
                out.append((ls, le))
    return out


def ventanas(cands, db):
    grupos = []
    for s, e in cands:
        if grupos and s - grupos[-1][1] < AGRUPAR:
            grupos[-1][1] = e
            grupos[-1][2].append((s, e))
        else:
            grupos.append([s, e, [(s, e)]])
    out = []
    for s, e, partes in grupos:
        a = punto_quieto(db, max(0.0, s - CONTEXTO), -1)
        b = punto_quieto(db, e + CONTEXTO, +1)
        if out and a <= out[-1]["b"]:
            out[-1]["b"] = max(out[-1]["b"], b)
            out[-1]["partes"] += partes
        else:
            out.append({"a": a, "b": b, "partes": partes})
    return out


def a_whisperx(w: dict) -> dict:
    """Word del schema Scribe -> formato word_segments de WhisperX."""
    d = {"word": w["text"], "start": float(w["start"]), "end": float(w["end"]),
         "score": math.exp(float(w.get("logprob", 0.0)))}
    if float(w.get("logprob", 0.0)) <= INTERP_LOGPROB:
        d["_interp"] = True
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-transcribe la voz que el ASR omitio")
    ap.add_argument("--edit-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="video (para el cache de energia)")
    ap.add_argument("--audio", type=Path, required=True, help="WAV mono 16 kHz del mic")
    ap.add_argument("--key", default="raw")
    ap.add_argument("--min-voz", type=float, default=0.25)
    ap.add_argument("--prompt-extra", default="", help="vocabulario propio de este video")
    ap.add_argument("--dry-run", action="store_true", help="solo listar las ventanas")
    ap.add_argument("--solo", action="append", default=[], metavar="A-B",
                    help="ventana manual en s de source (repetible); salta la deteccion y parte "
                         "del transcript ACTUAL, no del .orig (para corregir una ventana puntual)")
    ap.add_argument("--prompt", default=None,
                    help="initial_prompt completo (p.ej. con muletillas 'Eh... mmm, este' para que "
                         "Whisper no funda dos intentos de la misma frase en uno)")
    args = ap.parse_args()

    edit = args.edit_dir.resolve()
    tdir = edit / "transcripts"
    tp, orig = tdir / f"{args.key}.json", tdir / f"{args.key}.orig.json"
    if not orig.exists():
        shutil.copy2(tp, orig)
    data = json.loads((tp if args.solo else orig).read_text(encoding="utf-8"))
    words = sorted((w for w in data["words"] if V.es_palabra(w)), key=lambda w: float(w["start"]))

    db = np.load(edit / "verify" / f"{args.source.stem}.energy.npy")
    if args.solo:
        cands = []
        vents = [{"a": float(x.split("-")[0]), "b": float(x.split("-")[1]), "partes": []}
                 for x in args.solo]
    else:
        cands = candidatos(words, db, args.min_voz)
        vents = ventanas(cands, db)
    print(f"{len(cands)} tramos de voz sin words -> {len(vents)} ventanas "
          f"({sum(v['b'] - v['a'] for v in vents):.0f} s de audio)")

    def viejas(v):
        return [w for w in words if v["a"] <= (float(w["start"]) + float(w["end"])) / 2 < v["b"]]

    if args.dry_run:
        for v in vents:
            print(f"  {V.fmt_t(v['a'])}-{V.fmt_t(v['b'])}  voz sin words: "
                  f"{sum(e - s for s, e in v['partes']):.2f}s | "
                  f"{' '.join(V.txt(w) for w in viejas(v))[:110]}")
        return 0

    import torch  # antes que whisperx (DLLs de cudnn)
    import whisperx

    audio = whisperx.load_audio(str(args.audio))
    prompt = args.prompt or (DEFAULT_PROMPT + " " + args.prompt_extra).strip()
    t0 = time.time()
    model = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="es",
                                vad_method="silero", asr_options={"initial_prompt": prompt})
    for v in vents:
        clip = audio[int(v["a"] * SR):int(v["b"] * SR)]
        v["segs"] = model.transcribe(clip, batch_size=8, language="es")["segments"]
    del model
    torch.cuda.empty_cache()
    align_model, meta = whisperx.load_align_model(language_code="es", device="cuda")
    for v in vents:
        clip = audio[int(v["a"] * SR):int(v["b"] * SR)]
        ws = whisperx.align(v["segs"], align_model, meta, clip, "cuda",
                            return_char_alignments=False)["word_segments"] if v["segs"] else []
        ws = interpolate_missing(ws, v["b"] - v["a"])
        for w in ws:
            w["start"] = round(float(w["start"]) + v["a"], 3)
            w["end"] = round(float(w["end"]) + v["a"], 3)
        v["nuevas"] = ws
    print(f"re-transcrito en {time.time() - t0:.0f}s")

    # --- integrar ------------------------------------------------------------
    final = []
    dentro = set()
    lineas = ["# Re-transcripcion de voz sin words", ""]
    n_cambios = 0
    for v in vents:
        old = viejas(v)
        usar_nuevas = len(v["nuevas"]) >= len(old)
        t_old = " ".join(V.txt(w) for w in old)
        t_new = " ".join(w["word"].strip() for w in v["nuevas"])
        cambio = t_old != t_new
        n_cambios += cambio and usar_nuevas
        estado = "integrada" if usar_nuevas else "se queda la vieja (la nueva trae menos words)"
        if cambio:
            lineas += [f"## {V.fmt_t(v['a'])}-{V.fmt_t(v['b'])} — {estado}", "",
                       f"- antes: {t_old}", f"- ahora: {t_new}", ""]
        if usar_nuevas:
            dentro.update(id(w) for w in old)
            final += v["nuevas"]
    final += [a_whisperx(w) for w in words if id(w) not in dentro]
    final.sort(key=lambda w: float(w["start"]))

    payload = to_scribe_schema(final, " ".join(w["word"].strip() for w in final),
                               "es", float(data.get("language_probability", 1.0)),
                               float(data.get("audio_duration_secs", len(audio) / SR)))
    tp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (edit / ("retx_report_solo.md" if args.solo else "retx_report.md")).write_text("\n".join(lineas) + "\n", encoding="utf-8")
    n_w = sum(1 for w in payload["words"] if w["type"] == "word")
    print(f"{n_cambios} ventanas cambiaron | words: {len(words)} -> {n_w}")
    print(f"-> {tp}\n-> {edit / 'retx_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
