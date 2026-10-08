"""Build karaoke `subtitles.html` from a Scribe transcript + a (start, end) range.

Reads a transcript JSON (from helpers/transcribe.py) and a source time range,
groups words into "pills" of ~3-9 words (breaks on silence > 0.4s or punctuation),
and renders an HTML file compatible with the glass-composition-9x16 preset.

Usage:
    python helpers/build_karaoke.py \
        --transcript "<path>/transcripts/master.json" \
        --src-start 895.0 --src-end 955.0 \
        --template "<path>/presets/glass-composition-9x16/compositions/subtitles.html" \
        --output "<path>/slot/compositions/subtitles.html"

The output time of each word is `word.start - src_start` (clamped to [0, duration]).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PUNCT_BREAK = set(".!?")
SOFT_PUNCT = set(",;:")


def load_words_in_range(transcript_path: Path, src_start: float, src_end: float) -> list[dict]:
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    words = []
    for w in data.get("words", []):
        if w.get("type") != "word":
            continue
        ws = w.get("start")
        we = w.get("end")
        text = (w.get("text") or "").strip()
        if ws is None or we is None or not text:
            continue
        # Include if at least overlaps the range
        if we <= src_start or ws >= src_end:
            continue
        words.append({"start": float(ws), "end": float(we), "text": text})
    return words


def group_into_pills(
    words: list[dict],
    src_start: float,
    duration: float,
    silence_break: float = 0.45,
    max_per_pill: int = 7,
    min_per_pill: int = 2,
) -> list[dict]:
    """Group words into pill phrases. Returns list of pills, each with words list."""
    if not words:
        return []

    pills: list[list[dict]] = []
    current: list[dict] = []
    prev_end = None

    for w in words:
        gap = (w["start"] - prev_end) if prev_end is not None else 0.0

        # Decide if we should start a new pill BEFORE adding this word
        if current and (
            gap >= silence_break
            or len(current) >= max_per_pill
        ):
            # only flush if min_per_pill satisfied
            if len(current) >= min_per_pill:
                pills.append(current)
                current = []

        current.append(w)
        prev_end = w["end"]

        # Hard break after sentence-ending punctuation
        if w["text"][-1:] in PUNCT_BREAK and len(current) >= min_per_pill:
            pills.append(current)
            current = []
            prev_end = None

    if current:
        pills.append(current)

    # Convert to output-time pills
    out_pills = []
    for i, pwords in enumerate(pills, start=1):
        first = pwords[0]
        last = pwords[-1]
        show_at = max(0.0, first["start"] - src_start - 0.05)
        hide_at = min(duration, last["end"] - src_start + 0.35)
        out_pills.append({
            "id": f"pill-{i}",
            "show_at": round(show_at, 3),
            "hide_at": round(hide_at, 3),
            "words": [
                {
                    "text": w["text"],
                    "at": round(max(0.0, w["start"] - src_start), 3),
                }
                for w in pwords
            ],
        })
    return out_pills


SALIDA_PILL = 0.30   # duracion del tween de salida de la pill en el template (y -30, 0.30 s)


def corregir_encimados(pills: list[dict]) -> int:
    """La pill i debe terminar su salida antes de que entre la i+1 (mismo lugar en pantalla).

    Antes hide_at = fin de la ultima word + 0.35 sin mirar la siguiente: con cortes por
    max_per_pill o por puntuacion la siguiente entraba ~0.1 s despues y se veian dos
    pills encimadas (`check` lo reporta como content_overlap). Se respeta un minimo de
    0.25 s con la ultima word encendida.
    """
    n = 0
    for a, b in zip(pills, pills[1:]):
        tope = b["show_at"] - SALIDA_PILL
        if a["hide_at"] > tope:
            minimo = a["words"][-1]["at"] + 0.25
            a["hide_at"] = round(max(minimo, tope), 3)
            n += 1
    return n


def _norm(t: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ]", "", t)


def pills_de_frases(transcript_path: Path, ranges: list[tuple[float, float]], frases: list[str],
                    max_per_pill: int) -> tuple[list[dict], float, list[str]]:
    """Pills SOLO para frases clave (subtitulos selectivos, p.ej. una intro).

    Mapea todas las words del EDL al timeline de salida y busca cada frase como
    secuencia de tokens normalizados (sin acentos ni puntuacion). Una frase de mas de
    `max_per_pill` palabras se parte en varias pills. Devuelve las frases no encontradas.
    """
    words, off = [], 0.0
    for s, e in ranges:
        for w in load_words_in_range(transcript_path, s, e):
            words.append({"text": w["text"], "at": w["start"] - s + off,
                          "end": w["end"] - s + off, "tok": _norm(w["text"])})
        off += e - s
    toks = [w["tok"] for w in words]
    pills, faltan = [], []
    for frase in frases:
        ft = [t for t in (_norm(x) for x in frase.split()) if t]
        hit = next((i for i in range(len(toks) - len(ft) + 1) if toks[i:i + len(ft)] == ft), None)
        if hit is None:
            faltan.append(frase)
            continue
        ws = words[hit:hit + len(ft)]
        for k in range(0, len(ws), max_per_pill):
            trozo = ws[k:k + max_per_pill]
            pills.append({
                "show_at": round(max(0.0, trozo[0]["at"] - 0.05), 3),
                "hide_at": round(min(off, trozo[-1]["end"] + 0.35), 3),
                "words": [{"text": w["text"], "at": round(max(0.0, w["at"]), 3)} for w in trozo],
            })
    pills.sort(key=lambda p: p["show_at"])
    for i, p in enumerate(pills, start=1):
        p["id"] = f"pill-{i}"
    return pills, off, faltan


def render_pills_html(pills: list[dict]) -> str:
    """El chip CC va DENTRO de `.kar-words`, como primer item.

    Si se deja como hermano de `.kar-words` (que es como estaba antes), al envolver
    a varias lineas el chip queda solo arriba-izquierda y la pill se ve descentrada.
    Es el mismo bug que ya se arreglo a mano en `presets/glass-composition/`; el
    generador lo reintroducia en cada uso.
    """
    parts = []
    for p in pills:
        word_spans = "\n        ".join(
            f'<span class="kar-w" data-at="{w["at"]:.2f}">{w["text"]}</span>'
            for w in p["words"]
        )
        parts.append(
            f'<div class="kar-pill" id="{p["id"]}">\n'
            f'      <span class="kar-words">\n'
            f'        <span class="kar-cc">CC</span>\n'
            f'        {word_spans}\n'
            f'      </span>\n'
            f'    </div>'
        )
    return "\n\n    ".join(parts)


def render_phrases_js(pills: list[dict]) -> str:
    lines = ["const phrases = ["]
    for p in pills:
        lines.append(
            f'        {{ showAt: {p["show_at"]:.2f}, hideAt: {p["hide_at"]:.2f}, id: "{p["id"]}" }},'
        )
    lines.append("      ];")
    return "\n      ".join(lines)


def collect_pills_from_ranges(
    transcript_path: Path,
    ranges: list[tuple[float, float]],
    silence_break: float,
    max_per_pill: int,
) -> tuple[list[dict], float]:
    """Iterate multi-segment ranges and accumulate output-time pills.

    Returns (pills, total_duration). Each pill's word `at` and the pill's
    show_at/hide_at are in output-timeline seconds (not source).
    """
    all_pills: list[dict] = []
    out_offset = 0.0
    pill_idx = 0

    for seg_start, seg_end in ranges:
        seg_duration = seg_end - seg_start
        words = load_words_in_range(transcript_path, seg_start, seg_end)
        seg_pills = group_into_pills(
            words, seg_start, seg_duration,
            silence_break=silence_break,
            max_per_pill=max_per_pill,
        )
        # Shift this segment's times by out_offset (group_into_pills produced
        # times relative to seg_start; add the accumulated offset).
        for p in seg_pills:
            pill_idx += 1
            p["id"] = f"pill-{pill_idx}"
            p["show_at"] = round(p["show_at"] + out_offset, 3)
            p["hide_at"] = round(p["hide_at"] + out_offset, 3)
            for w in p["words"]:
                w["at"] = round(w["at"] + out_offset, 3)
        all_pills.extend(seg_pills)
        out_offset += seg_duration

    return all_pills, out_offset


def main() -> None:
    ap = argparse.ArgumentParser(description="Build karaoke subtitles.html from a transcript")
    ap.add_argument("--transcript", required=True, type=Path)
    ap.add_argument("--edl", type=Path,
                    help="EDL JSON; reads `ranges` for multi-segment mapping. "
                         "Mutually exclusive with --src-start/--src-end.")
    ap.add_argument("--src-start", type=float)
    ap.add_argument("--src-end", type=float)
    ap.add_argument("--template", required=True, type=Path,
                    help="Path to subtitles.html template with PILLS_PLACEHOLDER and PHRASES_PLACEHOLDER")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--silence-break", type=float, default=0.45)
    ap.add_argument("--max-per-pill", type=int, default=7)
    ap.add_argument("--frases", type=Path,
                    help="archivo con una frase clave por linea: solo esas frases llevan "
                         "subtitulo (el resto del tramo queda sin pill)")
    args = ap.parse_args()

    if args.edl:
        edl = json.loads(args.edl.read_text(encoding="utf-8"))
        ranges = [(float(r["start"]), float(r["end"])) for r in edl["ranges"]]
    else:
        if args.src_start is None or args.src_end is None:
            raise SystemExit("provide either --edl or both --src-start/--src-end")
        ranges = [(args.src_start, args.src_end)]

    if args.frases:
        frases = [l.strip() for l in args.frases.read_text(encoding="utf-8").splitlines() if l.strip()]
        pills, duration, faltan = pills_de_frases(args.transcript, ranges, frases, args.max_per_pill)
        if faltan:
            raise SystemExit("frases que NO aparecen en el transcript del tramo (revisar el texto "
                             "exacto) — no se escribio nada:\n  - " + "\n  - ".join(faltan))
    else:
        pills, duration = collect_pills_from_ranges(
            args.transcript, ranges,
            silence_break=args.silence_break,
            max_per_pill=args.max_per_pill,
        )
    if not pills:
        raise SystemExit("no words found in the given range(s)")
    n_fix = corregir_encimados(pills)
    if n_fix:
        print(f"  {n_fix} pills acortadas para no encimarse con la siguiente")
    total_words = sum(len(p["words"]) for p in pills)
    print(f"ranges={len(ranges)}  words={total_words}  pills={len(pills)}  duration={duration:.2f}s")

    template = args.template.read_text(encoding="utf-8")
    pills_html = render_pills_html(pills)
    phrases_js = render_phrases_js(pills)

    # Cada sustitucion se VERIFICA. Antes, si el template no traia el marcador,
    # `.replace()` no hacia nada, el helper igual imprimia "pills=N ... wrote <path>"
    # y escribia el template tal cual: subtitulos de OTRO video, sin un solo error.
    problems: list[str] = []

    out = template
    for marker, repl, what in (
        ("<!-- PILLS_PLACEHOLDER -->", pills_html, "el markup de las pills"),
        ("// PHRASES_PLACEHOLDER", phrases_js, "el array `phrases` del timeline"),
    ):
        n = out.count(marker)
        if n == 0:
            problems.append(f"falta el marcador `{marker}` (ahi va {what})")
        elif n > 1:
            # p.ej. si el template lo menciona en un comentario de cabecera: la
            # sustitucion caeria en esa ocurrencia y el contenido quedaria comentado
            problems.append(
                f"el marcador `{marker}` aparece {n} veces y debe aparecer 1 sola "
                "(¿esta tambien dentro de un comentario del template?)")
        else:
            out = out.replace(marker, repl, 1)

    # La duracion se sustituye por PATRON, no contra un literal: el template puede
    # traer cualquier numero (antes se buscaba `60.0` y con `54.100` fallaba mudo).
    out, n_dur = re.subn(r'(data-duration=")[\d.]+(")',
                         rf'\g<1>{duration:.3f}\g<2>', out, count=1)
    if n_dur != 1:
        problems.append('no encontre `data-duration="<numero>"` en el root del template')

    out, n_pad = re.subn(r'(tl\.to\(\{\},\s*\{\s*duration:\s*)[\d.]+(\s*\},\s*0\);)',
                         rf'\g<1>{duration:.3f}\g<2>', out, count=1)
    if n_pad != 1:
        problems.append("no encontre el padding del timeline `tl.to({}, { duration: <numero> }, 0);`")

    if problems:
        raise SystemExit(
            "El template no es sustituible — NO se escribio nada:\n  - "
            + "\n  - ".join(problems)
            + f"\n\ntemplate: {args.template}\n"
            "Un template valido tiene los dos marcadores y CERO pills hardcodeadas.\n"
            "Si el preset se piso con la version concreta de un video, restaurar los\n"
            "marcadores (ver el gotcha de build_karaoke.py en CLAUDE.md)."
        )

    # Comprobaciones de salida.
    if "PLACEHOLDER" in out:
        raise SystemExit(
            "quedo algun PLACEHOLDER sin sustituir en la salida — NO se escribio nada.\n"
            f"template: {args.template}")
    n_written = out.count('class="kar-pill"')
    if n_written != len(pills):
        raise SystemExit(
            f"se calcularon {len(pills)} pills pero el HTML resultante tiene {n_written} — "
            "el template probablemente traia pills hardcodeadas ademas del marcador"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out, encoding="utf-8")
    print(f"wrote {args.output}  ({n_written} pills, data-duration={duration:.3f})")


if __name__ == "__main__":
    main()
