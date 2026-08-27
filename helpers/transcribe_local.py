"""Transcripción local con WhisperX (GPU) — reemplazo drop-in de Scribe.

Extrae audio mono 16kHz con ffmpeg, transcribe con faster-whisper (large-v3)
y alinea word-level con wav2vec2 (forced alignment). Escribe un JSON con el
MISMO schema que ElevenLabs Scribe (words con type word/spacing, speaker_id,
logprob) para que pack_transcripts.py y el resto del pipeline funcionen igual.

Requiere el venv separado .venv-asr (Python 3.12 + torch CUDA + whisperx):
    .venv-asr\\Scripts\\python.exe helpers/transcribe_local.py <video> --edit-dir <edit>

DEFAULT del pipeline desde 2026-07-26 (decisión de Jaime tras A/B vs Scribe).
Escribe <stem>.json igual que transcribe.py — drop-in, mismo cache. Usar
--suffix para escribir <stem>.<suffix>.json sin pisar un transcript existente
(ej. --suffix whisperx para A/B contra Scribe).

Notas vs Scribe:
- Sin diarización: speaker_id fijo "speaker_0" (videos de un solo speaker).
- logprob = log(score) del alignment (score 0-1 de wav2vec2), no es comparable
  1:1 con el logprob de Scribe pero sirve igual como señal de confianza.
- Palabras que el aligner no ancla (números, anglicismos raros) se interpolan
  entre sus vecinas y llevan logprob -9.99 como marca.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Windows sin Developer Mode no permite symlinks — HF cache debe copiar archivos
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# Vocabulario que Scribe transcribe mal — bias del decoder vía initial_prompt
DEFAULT_PROMPT = (
    "Video en español sobre automatización con IA. Menciona: VisiumAI, "
    "Claude Code, n8n, Hyperframes, Airtable, Supabase, Vapi, Retell, "
    "Pipedrive, CRM, YouTube, workflows, agentes."
)

# Marca para palabras sin alignment (timestamps interpolados)
INTERP_LOGPROB = -9.99

# Códigos de idioma estilo Scribe (ISO 639-2) desde los de Whisper (639-1)
LANG_MAP = {"es": "spa", "en": "eng", "pt": "por", "fr": "fra", "de": "deu", "it": "ita"}


def extract_audio(video_path: Path, dest: Path) -> None:
    # Mismo formato que transcribe.py: mono 16kHz PCM
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def interpolate_missing(words: list[dict], audio_dur: float) -> list[dict]:
    """Asigna timestamps a palabras que el aligner no pudo anclar.

    Reparte el hueco entre la vecina anterior y la siguiente con timestamps
    reales, proporcional al largo del texto de cada palabra sin anclar.
    """
    n = len(words)
    i = 0
    while i < n:
        if words[i].get("start") is not None:
            i += 1
            continue
        # Bloque contiguo [i, j) sin timestamps
        j = i
        while j < n and words[j].get("start") is None:
            j += 1
        left = words[i - 1]["end"] if i > 0 else 0.0
        right = words[j]["start"] if j < n else audio_dur
        total_chars = sum(max(len(w["word"]), 1) for w in words[i:j]) or 1
        span = max(right - left, 0.02 * (j - i))
        t = left
        for w in words[i:j]:
            frac = max(len(w["word"]), 1) / total_chars
            w["start"] = round(t, 3)
            t = min(t + span * frac, right)
            w["end"] = round(t, 3)
            w["_interp"] = True
        i = j
    return words


def to_scribe_schema(
    word_segments: list[dict],
    full_text: str,
    language: str,
    lang_prob: float,
    audio_dur: float,
) -> dict:
    """Convierte word_segments de WhisperX al schema exacto de Scribe.

    Genera tokens 'spacing' entre palabras consecutivas — pack_transcripts.py
    los usa para medir las pausas, deben cubrir el gap completo.
    """
    words_out: list[dict] = []
    prev_end: float | None = None

    for w in word_segments:
        start = float(w["start"])
        end = float(w["end"])
        if w.get("_interp"):
            logprob = INTERP_LOGPROB
        else:
            score = float(w.get("score", 1.0))
            logprob = round(math.log(max(score, 1e-6)), 6)

        if prev_end is not None:
            # spacing cubre TODO el gap entre palabras (así lo hace Scribe)
            words_out.append({
                "text": " ",
                "start": round(prev_end, 3),
                "end": round(start, 3),
                "type": "spacing",
                "speaker_id": "speaker_0",
                "logprob": 0.0,
            })

        words_out.append({
            "text": w["word"].strip(),
            "start": round(start, 3),
            "end": round(end, 3),
            "type": "word",
            "speaker_id": "speaker_0",
            "logprob": logprob,
        })
        prev_end = end

    return {
        "language_code": LANG_MAP.get(language, language),
        "language_probability": round(float(lang_prob), 4),
        "text": full_text,
        "words": words_out,
        "transcription_id": f"whisperx-{uuid.uuid4().hex[:12]}",
        "audio_duration_secs": round(audio_dur, 4),
    }


def transcribe_whisperx(
    audio_path: Path,
    model_name: str,
    language: str | None,
    prompt: str,
    batch_size: int,
    compute_type: str,
    verbose: bool = True,
) -> dict:
    import torch  # primero: registra torch\lib en el DLL search path (cudnn para ctranslate2)
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        compute_type = "int8"
        print("  AVISO: CUDA no disponible, corriendo en CPU (lento)", flush=True)

    audio = whisperx.load_audio(str(audio_path))
    audio_dur = len(audio) / 16000.0

    if verbose:
        print(f"  audio: {audio_dur:.1f}s | device: {device} | modelo: {model_name}", flush=True)

    t0 = time.time()
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
        vad_method="silero",  # sin dependencia de HF auth (pyannote es gated)
        asr_options={"initial_prompt": prompt} if prompt else None,
    )
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    detected_lang = result["language"]
    if verbose:
        print(f"  transcrito en {time.time() - t0:.1f}s (lang={detected_lang})", flush=True)

    # Liberar el modelo whisper antes de cargar el de alignment (VRAM)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    t1 = time.time()
    align_model, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )
    if verbose:
        print(f"  alineado en {time.time() - t1:.1f}s", flush=True)

    del align_model
    gc.collect()
    torch.cuda.empty_cache()

    full_text = " ".join(s["text"].strip() for s in result["segments"]).strip()
    word_segments = interpolate_missing(aligned["word_segments"], audio_dur)
    n_interp = sum(1 for w in word_segments if w.get("_interp"))
    if verbose and n_interp:
        print(f"  {n_interp} palabras sin alignment (timestamps interpolados)", flush=True)

    return to_scribe_schema(word_segments, full_text, detected_lang, 1.0, audio_dur)


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcripción local con WhisperX (schema Scribe)")
    ap.add_argument("video", type=Path, help="Ruta al video")
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Directorio edit (default: <video_parent>/edit)")
    ap.add_argument("--language", type=str, default="es",
                    help="Código ISO del idioma ('es'). Usar 'auto' para autodetectar.")
    ap.add_argument("--model", type=str, default="large-v3",
                    help="Modelo whisper (large-v3, large-v3-turbo, medium...)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--compute-type", type=str, default="float16")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                    help="initial_prompt con vocabulario ('' para desactivar)")
    ap.add_argument("--suffix", type=str, default="",
                    help="Sufijo del archivo: <stem>.<suffix>.json (para A/B sin pisar cache)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Ruta de salida (default: transcripts/<stem>.json)")
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video no encontrado: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{video.stem}.{args.suffix}" if args.suffix else video.stem
    out_path = args.output or (transcripts_dir / f"{stem}.json")

    if out_path.exists():
        print(f"cached: {out_path.name}")
        return

    language = None if args.language == "auto" else args.language

    print(f"  extrayendo audio de {video.name}", flush=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        payload = transcribe_whisperx(
            audio, args.model, language, args.prompt,
            args.batch_size, args.compute_type,
        )

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    n_words = sum(1 for w in payload["words"] if w["type"] == "word")
    print(f"  guardado: {out_path.name} en {time.time() - t0:.1f}s")
    print(f"    words: {n_words}")


if __name__ == "__main__":
    main()
