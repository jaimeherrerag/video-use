"""Auditor del entregable: transcribe el video final y lo compara con lo que deberia decir.

Por que existe (short de REA, 2026-10-12): los checks sobre el EDL miden el source,
no lo que de verdad suena en el video final. Ahi se colaron tres palabras cortadas
("Anything", "estrellas", "en GitHub") que solo encontro Jaime al ver la vista
previa. Transcribir el entregable completo y compararlo con el texto esperado lo
habria detectado en la primera ronda.

Texto esperado = las words del transcript que caen dentro de cada range, en orden
(lo mismo que alimenta el karaoke). Se compara por tokens normalizados: sin
acentos ni puntuacion, y cualquier numero (en cifra o en palabras) vale <num>,
para que "150.000" y "ciento cincuenta mil" no cuenten como diferencia.

  - ERROR falta-en-juntura: falta una palabra esperada junto a un corte (el
          error tipico: el corte se la comio).
  - WARN  falta: falta una palabra a mitad de un range (puede ser el ASR).
  - WARN  sobra / cambia: el video dice algo distinto de lo esperado.
Las variaciones de una sola palabra parecida ("cloud"/"claude") se ignoran.

El reporte lo juzga despues el subagente auditor-edicion (Haiku): separa errores
reales de ruido del ASR y hace la lectura editorial del texto final.

Uso (en .venv-asr, desde video-use/):
    .venv-asr/Scripts/python.exe helpers/audit_entregable.py --edit-dir "<edit>" [--video preview_final.mp4]
Salida: <edit>/verify/entregable_report.md y .json. Exit 1 si hay ERROR.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bordes import tokens  # noqa: E402
from verify_edl import cargar, es_palabra, fmt_t, txt  # noqa: E402

JUNTURA_PALABRAS = 2      # las N primeras / ultimas palabras de un range cuentan como "junto al corte"
SIMILAR_MIN = 0.6         # una sola palabra parecida a la esperada = variacion del ASR, no error


def esperado(edl_path: Path):
    """[(token, idx_range, pos_en_range, n_en_range, texto_original)] en orden de salida."""
    edl, _, transcripts = cargar(edl_path)
    out = []
    for i, r in enumerate(edl.get("ranges", [])):
        s, e = float(r["start"]), float(r["end"])
        ws = [w for w in transcripts.get(r.get("source"), [])
              if es_palabra(w) and s - 0.02 <= float(w["start"]) < e]
        toks = []
        for w in ws:
            for t in tokens(txt(w)):
                if toks and t == "<num>" and toks[-1][0] == "<num>":
                    continue
                toks.append((t, txt(w)))
        for k, (t, orig) in enumerate(toks):
            out.append((t, i, k, len(toks), orig))
    return out


def transcribir(video: Path):
    """[(token, t_salida, texto)] del video final, con faster-whisper large-v3 y word timestamps."""
    import torch  # primero: registra torch\\lib en el DLL search path (cudnn)
    from faster_whisper import WhisperModel
    from transcribe_local import DEFAULT_PROMPT
    import whisperx
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = WhisperModel("large-v3", device=dev, compute_type="float16" if dev == "cuda" else "int8")
    audio = whisperx.load_audio(str(video))
    segs, _ = m.transcribe(audio, language="es", initial_prompt=DEFAULT_PROMPT, vad_filter=True,
                           word_timestamps=True, beam_size=5, condition_on_previous_text=False)
    out = []
    for sg in segs:
        for w in sg.words or []:
            for t in tokens(w.word):
                if out and t == "<num>" and out[-1][0] == "<num>":
                    continue
                out.append((t, float(w.start), w.word.strip()))
    return out


def comparar(esp, got):
    a, b = [x[0] for x in esp], [x[0] for x in got]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    hallazgos = []

    def t_cerca(j):
        if not got:
            return 0.0
        return got[min(max(j, 0), len(got) - 1)][1]

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        ea, gb = esp[i1:i2], got[j1:j2]
        txt_e = " ".join(x[4] for x in ea)
        txt_g = " ".join(x[2] for x in gb)
        if op == "replace" and len(ea) == 1 and len(gb) == 1 and \
                difflib.SequenceMatcher(None, ea[0][0], gb[0][0]).ratio() >= SIMILAR_MIN:
            continue                                   # "cloud"/"claude": variacion del ASR
        en_juntura = any(k < JUNTURA_PALABRAS or k >= n - JUNTURA_PALABRAS for _, _, k, n, _ in ea)
        rango = ea[0][1] if ea else None
        if op == "delete":
            sev = "ERROR" if en_juntura else "WARN"
            tipo = "falta-en-juntura" if en_juntura else "falta"
            hallazgos.append(dict(sev=sev, tipo=tipo, range=rango, out=round(t_cerca(j1), 2),
                                  esperado=txt_e, video="", msg=f"no se oye: \"{txt_e}\""))
        elif op == "insert":
            hallazgos.append(dict(sev="WARN", tipo="sobra", range=None, out=round(t_cerca(j1), 2),
                                  esperado="", video=txt_g, msg=f"el video dice de mas: \"{txt_g}\""))
        else:
            sev = "ERROR" if en_juntura and len(gb) < len(ea) else "WARN"
            hallazgos.append(dict(sev=sev, tipo="cambia" if sev == "WARN" else "falta-en-juntura", range=rango,
                                  out=round(t_cerca(j1), 2), esperado=txt_e, video=txt_g,
                                  msg=f"esperado \"{txt_e}\" / se oye \"{txt_g}\""))
    return hallazgos, sm.ratio()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compara lo que dice el entregable con lo que deberia decir")
    ap.add_argument("--edit-dir", type=Path, required=True)
    ap.add_argument("--video", type=Path, default=None, help="default: <edit>/preview_final.mp4")
    args = ap.parse_args()
    edit = args.edit_dir.resolve()
    video = (args.video or edit / "preview_final.mp4").resolve()
    if not video.exists():
        sys.exit(f"no existe el entregable: {video}")

    t0 = time.time()
    esp = esperado(edit / "edl.json")
    got = transcribir(video)
    hallazgos, parecido = comparar(esp, got)
    texto_video = " ".join(x[2] for x in got)

    vdir = edit / "verify"
    vdir.mkdir(exist_ok=True)
    (vdir / "entregable_report.json").write_text(json.dumps(
        dict(video=str(video), parecido=round(parecido, 4), hallazgos=hallazgos, texto_video=texto_video),
        ensure_ascii=False, indent=2), encoding="utf-8")
    lineas = [f"# Auditoría del entregable — {edit.parent.name}", "",
              f"`{video.name}` · {len(esp)} tokens esperados · {len(got)} oídos · parecido {parecido:.1%} · "
              f"{len(hallazgos)} diferencias · {time.time() - t0:.0f} s", ""]
    for h in hallazgos:
        r = f" r{h['range']}" if h["range"] is not None else ""
        lineas.append(f"- **{h['sev']}** `{h['tipo']}`{r} · video {fmt_t(h['out'])} — {h['msg']}")
    lineas += ["", "## Texto del video (ASR)", "", texto_video]
    (vdir / "entregable_report.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    for h in hallazgos:
        print(f"  {h['sev']:5} {h['tipo']:17} video {fmt_t(h['out'])} | {h['msg'][:90]}")
    n_err = sum(h["sev"] == "ERROR" for h in hallazgos)
    print(f"\nparecido {parecido:.1%} | {n_err} ERROR | {sum(h['sev'] == 'WARN' for h in hallazgos)} WARN "
          f"({time.time() - t0:.0f} s) -> {vdir / 'entregable_report.md'}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
