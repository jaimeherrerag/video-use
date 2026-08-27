"""Render a video from an EDL.

Implements the HEURISTICS render pipeline in the correct order:

  1. Per-segment extract with color grade + 30ms audio fades baked in
  2. Lossless -c copy concat into base.mp4
  3. If overlays or subtitles: single filter graph that overlays animations
     (with PTS shift so frame 0 lands at the overlay window start)
     and applies `subtitles` filter LAST → final.mp4

Optionally builds a master SRT from the per-source transcripts + EDL
output-timeline offsets, applies the proven force_style (2-word
UPPERCASE chunks, Helvetica 18 Bold, MarginV=35).

Usage:
    python helpers/render.py <edl.json> -o final.mp4
    python helpers/render.py <edl.json> -o preview.mp4 --preview
    python helpers/render.py <edl.json> -o final.mp4 --build-subtitles
    python helpers/render.py <edl.json> -o final.mp4 --no-subtitles
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    from grade import get_preset, auto_grade_for_clip  # same directory
except Exception:
    def get_preset(name: str) -> str:
        return ""

    def auto_grade_for_clip(video, start=0.0, duration=None, verbose=False):  # type: ignore
        return "eq=contrast=1.03:saturation=0.98", {}


# -------- Subtitle style (bold-overlay, proven at 1920×1080 and 1080×1920) --
#
# MarginV is NOT taste — it is a platform safe-zone rule.
# TikTok / IG Reels / Shorts UI (caption, username, music, right-rail actions)
# covers roughly the bottom ~25–30% of a 1080×1920 frame. Captions placed near
# the bottom edge get clipped or obscured by the UI. libass auto-scales the
# render canvas relative to PlayResY=288, so MarginV=90 lands the caption
# baseline roughly 30% up from the bottom on any aspect — clear of the UI on
# every major vertical-video platform. Do not drop this below ~75 without a
# specific reason.
SUB_FORCE_STYLE = (
    "FontName=Helvetica,FontSize=18,Bold=1,"
    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,"
    "BorderStyle=1,Outline=2,Shadow=0,"
    "Alignment=2,MarginV=20"
)

# -------- Helpers ------------------------------------------------------------


def run(cmd: list[str], quiet: bool = False) -> None:
    if not quiet:
        print(f"  $ {' '.join(str(c) for c in cmd[:6])}{' …' if len(cmd) > 6 else ''}")
    subprocess.run(cmd, check=True)


def resolve_grade_filter(grade_field: str | None) -> str:
    """The EDL's 'grade' field can be a preset name, a raw ffmpeg filter, or 'auto'.

    Returns the filter string to embed into the per-segment -vf chain.
    For 'auto', returns the sentinel "__AUTO__" which is resolved per-segment.
    """
    if not grade_field:
        return ""
    if grade_field == "auto":
        return "__AUTO__"
    # Preset names are short identifiers, filter strings contain '=' or ','.
    if re.fullmatch(r"[a-zA-Z0-9_\-]+", grade_field):
        try:
            return get_preset(grade_field)
        except KeyError:
            print(f"warning: unknown preset '{grade_field}', using as raw filter")
            return grade_field
    return grade_field


def resolve_path(maybe_path: str, base: Path) -> Path:
    """Resolve a path that may be absolute or relative to `base`."""
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


# -------- HDR → SDR tone mapping (HLG / PQ sources) --------------------------
#
# iPhone defaults to HLG HDR in Rec.2020 (and many mirrorless cameras ship PQ).
# If the source is HDR and we only downconvert bit depth (yuv420p10le → yuv420p)
# without tone-mapping, the output is 8-bit but still carries HLG/PQ transfer
# metadata. Players that honor the metadata (screen recorders, most social
# upload re-encodes) interpret 8-bit values in an HDR container and the result
# looks oversaturated / blown out. QuickTime on macOS can hide this locally —
# screen recording and uploaded renders cannot.
#
# Fix: detect HDR via color_transfer and prepend a zscale+tonemap chain to the
# vf graph so the output is clean Rec.709 SDR.

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ (HDR10) and HLG

TONEMAP_CHAIN = (
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,"
    "format=yuv420p"
)


def is_hdr_source(video: Path) -> bool:
    """Return True if the source uses a PQ or HLG transfer function."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() in HDR_TRANSFERS
    except subprocess.CalledProcessError:
        return False


def is_portrait_source(video: Path) -> bool:
    """Return True if the video's height > width (portrait / vertical)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True,
        )
        w, h = map(int, out.stdout.strip().split(","))
        return h > w
    except Exception:
        return False


# -------- Codec de video (x264 CPU por default, NVENC GPU con --nvenc) -------


def nvenc_available() -> bool:
    """True si este ffmpeg trae el encoder h264_nvenc (GPU NVIDIA + drivers)."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True,
        )
    except Exception:
        return False
    return "h264_nvenc" in out.stdout


def video_codec_args(
    *, nvenc: bool, draft: bool = False, preview: bool = False, high_quality: bool = False
) -> list[str]:
    """Bloque de códec de video para los encodes de segmento.

    El entregable final es quality-first: x264 slow/crf17 sigue siendo mejor
    calidad-por-bit que NVENC a bitrate bajo → sigue siendo el default de
    --high-quality y del render sin flags.

    NVENC usa **H.264**, no HEVC (medido 2026-08-04: mismo s/clip y mismo tamaño
    que hevc_nvenc, pero H.264 evita dos fallos silenciosos de HEVC):
      - Chrome headless NO decodifica HEVC → un base HEVC embebido en una
        composición HyperFrames sale NEGRO sin warning. `--preview` genera
        `base_preview.mp4`, que es justo el archivo que se copia a
        `slot_hf/assets/base.mp4`.
      - `-ss` no busca bien en HEVC → extraer fragmentos de un preview para
        re-transcribir devuelve audio basura/alucinado.
    `-cq` controla calidad (menor = mejor); `-b:v 0` deja a CQ como único driver.
    CQ 1-2 puntos por debajo del equivalente HEVC: H.264 es menos eficiente.
    """
    if nvenc:
        if draft:
            nv_preset, cq = "p2", "30"
        elif preview:
            nv_preset, cq = "p5", "25"
        elif high_quality:
            nv_preset, cq = "p7", "18"
        else:
            nv_preset, cq = "p6", "20"
        return [
            "-c:v", "h264_nvenc", "-preset", nv_preset, "-tune", "hq",
            "-rc", "vbr", "-cq", cq, "-b:v", "0", "-pix_fmt", "yuv420p",
        ]
    # x264 (CPU) — quality ladder
    if draft:
        preset, crf = "ultrafast", "28"
    elif preview:
        preset, crf = "medium", "22"
    elif high_quality:
        preset, crf = "slow", "17"
    else:
        preset, crf = "fast", "20"
    return ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p"]


# -------- Per-segment extraction (Rule 2 + Rule 3) --------------------------


def censor_chain(regions: list) -> str:
    """Blur rectangular regions (source-pixel coords) via split/crop/boxblur/overlay.

    Returns a filtergraph fragment with one unlabeled input and output, safe to
    embed in a -vf chain or after [0:v] in a filter_complex. Regions are blurred
    hard enough to make on-screen text (IPs, keys) unreadable.
    """
    parts = []
    for i, r in enumerate(regions):
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        parts.append(
            f"split[cb{i}][cr{i}];"
            f"[cr{i}]crop={w}:{h}:{x}:{y},"
            # radius capped at half the region size (boxblur hard limit); power 3
            # keeps small text unreadable even with a small radius
            f"boxblur=luma_radius=min(min(w\\,h)/2-1\\,12):luma_power=3:"
            f"chroma_radius=min(min(cw\\,ch)/2-1\\,6):chroma_power=3[cx{i}];"
            f"[cb{i}][cx{i}]overlay={x}:{y}"
        )
    return ",".join(parts)


def extract_segment(
    source: Path,
    seg_start: float,
    duration: float,
    grade_filter: str,
    out_path: Path,
    preview: bool = False,
    draft: bool = False,
    reframe: dict | None = None,
    fps: int = 24,
    high_quality: bool = False,
    nvenc: bool = False,
    audio_filter: str | None = None,
    censor: list | None = None,
) -> None:
    """Extract a cut range as its own MP4 with grade + 30ms audio fades baked in.

    `-ss` before `-i` for fast accurate seeking. Scale to 1080p from 4K.
    Portrait sources (height > width) are scaled by height to preserve orientation.

    Quality ladder:
      - final (default): 1080p libx264 fast CRF 20
      - preview:         1080p libx264 medium CRF 22 (evaluable for QC)
      - draft:           720p libx264 ultrafast CRF 28 (cut-point check only)

    Optional `reframe` dict (e.g. for 16:9 → 9:16 shorts):
      {
        "src_crop": { "x": int, "y": int, "w": int, "h": int },  # window in source pixels
        "out_size": { "w": 1080, "h": 1920 },                   # target output size
        "fit":      "blur-bg",                                  # only mode for now
        "blur_sigma": 20                                        # optional
      }
    When set, the filter graph splits the input into a blureado full-frame
    background and a cropped/scaled foreground centered vertically.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vcodec = video_codec_args(
        nvenc=nvenc, draft=draft, preview=preview, high_quality=high_quality
    )

    # 30ms audio fades at both edges (Rule 3) — prevent pops.
    # `apad` + `-shortest` (below) trim audio to EXACTLY the frame-quantized video
    # duration so per-segment A/V durations match. Without this, video (discrete
    # frames) and audio (exact samples) drift a few ms/segment, accumulating into
    # progressive audio desync across a long concat. See CLAUDE.md.
    fade_out_start = max(0.0, duration - 0.03)
    af = f"afade=t=in:st=0:d=0.03,afade=t=out:st={fade_out_start:.3f}:d=0.03,apad"
    # Filtro de audio opcional (ej. denoise) ANTES de fades/apad, en la etapa PCM.
    # OJO: afftdn aquí junto a un -filter_complex de video (reframe) deadlockea
    # ffmpeg 8.0.1 — para EDLs con reframe, denoisear el source completo en un
    # pase (source_dn.mov) en vez de usar este campo. Ver CLAUDE.md.
    if audio_filter:
        af = f"{audio_filter},{af}"

    # Accurate input seek (`-ss` before `-i`, with ffmpeg's default accurate_seek):
    # seeks to the keyframe, decodes, and discards up to seg_start so BOTH audio
    # and video begin exactly at seg_start. Do NOT combine with an output `-ss`
    # (hybrid seek): the input seek lands on a keyframe before seg_start and the
    # output `-ss` then measures from there, starting the content early and
    # clipping speech off the END of every segment. Render at the source's native
    # fps (`--fps 60`) to avoid frame-decimation lip-sync artifacts. See CLAUDE.md.
    if reframe is None:
        # Legacy path: simple scale + grade, no reframe.
        portrait = is_portrait_source(source)
        if draft:
            scale = "scale=-2:1280" if portrait else "scale=1280:-2"
        else:
            scale = "scale=-2:1920" if portrait else "scale=1920:-2"

        vf_parts: list[str] = []
        # Censor regions use SOURCE-pixel coords: apply before any scale.
        if censor:
            vf_parts.append(censor_chain(censor))
        if is_hdr_source(source):
            vf_parts.append(TONEMAP_CHAIN)
        vf_parts.append(scale)
        if grade_filter:
            vf_parts.append(grade_filter)
        vf = ",".join(vf_parts)

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{seg_start:.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-vf", vf,
            "-af", af,
            *vcodec, "-r", str(fps),
            # PCM lossless: el único encode AAC ocurre al final (loudnorm). Ver FINAL_AAC_BITRATE.
            "-c:a", "pcm_s16le", "-ar", "48000",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return

    # ----- Reframe path: blur-bg (9:16 shorts) or crop-scale (16:9 top-crop) -----
    fit = reframe.get("fit", "blur-bg")
    if fit not in ("blur-bg", "crop-scale", "crop-pad"):
        raise ValueError(f"unsupported reframe.fit: {fit}")

    src_crop = reframe["src_crop"]
    cx = int(src_crop["x"]); cy = int(src_crop["y"])
    cw = int(src_crop["w"]); ch = int(src_crop["h"])
    out_size = reframe.get("out_size", {"w": 1080, "h": 1920})
    out_w = int(out_size["w"]); out_h = int(out_size["h"])

    # Optional HDR tonemap applied before split so both layers are SDR.
    prelude = TONEMAP_CHAIN + "," if is_hdr_source(source) else ""
    # Censor regions (source-pixel coords) go first, before crop/scale.
    if censor:
        prelude = censor_chain(censor) + "," + prelude
    grade_suffix = f",{grade_filter}" if grade_filter else ""

    if fit == "crop-scale":
        # Crop source window and scale to fill output. Minor geometric distortion accepted.
        fc = (
            f"[0:v]{prelude}crop={cw}:{ch}:{cx}:{cy},"
            f"scale={out_w}:{out_h}:flags=lanczos{grade_suffix}[outv]"
        )
    elif fit == "crop-pad":
        # Crop source window and center it on a black canvas — no zoom, no stretch.
        # (e.g. hide browser chrome: crop y=115..1080 and letterbox 57/58px top/bottom)
        fc = (
            f"[0:v]{prelude}crop={cw}:{ch}:{cx}:{cy},"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black{grade_suffix}[outv]"
        )
    else:  # blur-bg
        blur_sigma = int(reframe.get("blur_sigma", 20))
        fg_y_offset = int(reframe.get("fg_y_offset_px", 0))
        # Background: scale full source to cover output, gaussian blur.
        # Foreground: crop window, scale to output width, overlay vertically centered.
        y_expr = "(H-h)/2" if fg_y_offset == 0 else f"(H-h)/2{fg_y_offset:+d}"
        fc = (
            f"[0:v]{prelude}split=2[fg_src][bg_src];"
            f"[bg_src]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},gblur=sigma={blur_sigma}[bg];"
            f"[fg_src]crop={cw}:{ch}:{cx}:{cy},scale={out_w}:-2{grade_suffix}[fg];"
            f"[bg][fg]overlay=x=0:y={y_expr}:format=auto[outv]"
        )

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{seg_start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-filter_complex", fc,
        "-map", "[outv]",
        "-map", "0:a",
        "-af", af,
        *vcodec, "-r", str(fps),
        # PCM lossless: el único encode AAC ocurre al final (loudnorm). Ver FINAL_AAC_BITRATE.
        "-c:a", "pcm_s16le", "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def extract_all_segments(
    edl: dict,
    edit_dir: Path,
    preview: bool,
    draft: bool = False,
    fps: int = 24,
    high_quality: bool = False,
    nvenc: bool = False,
) -> list[Path]:
    """Extract every EDL range into edit_dir/clips_graded/seg_NN.mp4.
    Returns the ordered list of segment paths.

    If the EDL `grade` is "auto", analyze each segment range with
    `auto_grade_for_clip` and apply a per-segment subtle correction.
    Otherwise, apply the same preset/raw filter to every segment.
    """
    resolved = resolve_grade_filter(edl.get("grade"))
    is_auto = resolved == "__AUTO__"
    clips_dir = edit_dir / (
        "clips_draft" if draft else ("clips_preview" if preview else "clips_graded")
    )
    clips_dir.mkdir(parents=True, exist_ok=True)

    ranges = edl["ranges"]
    sources = edl["sources"]
    audio_filter = edl.get("audio_filter")

    seg_paths: list[Path] = []
    print(f"extracting {len(ranges)} segment(s) → {clips_dir.name}/")
    if is_auto:
        print("  (auto-grade per segment: analyzing each range)")
    if audio_filter:
        print(f"  audio_filter: {audio_filter}")
    for i, r in enumerate(ranges):
        src_name = r["source"]
        src_path = resolve_path(sources[src_name], edit_dir)
        start = float(r["start"])
        end = float(r["end"])
        duration = end - start
        out_path = clips_dir / f"seg_{i:02d}_{src_name}.mp4"

        if is_auto:
            seg_filter, _stats = auto_grade_for_clip(src_path, start=start, duration=duration, verbose=False)
        else:
            seg_filter = resolved

        note = r.get("beat") or r.get("note") or ""
        reframe = r.get("reframe")
        rf_tag = f"  reframe={reframe['fit']}" if reframe else ""
        print(f"  [{i:02d}] {src_name}  {start:7.2f}-{end:7.2f}  ({duration:5.2f}s)  {note}{rf_tag}")
        if is_auto:
            print(f"        grade: {seg_filter or '(none)'}")
        extract_segment(
            src_path, start, duration, seg_filter, out_path,
            preview=preview, draft=draft, reframe=reframe,
            fps=fps, high_quality=high_quality, nvenc=nvenc,
            audio_filter=audio_filter, censor=r.get("censor"),
        )
        seg_paths.append(out_path)

    return seg_paths


# -------- Lossless concat ----------------------------------------------------


def concat_segments(segment_paths: list[Path], out_path: Path, edit_dir: Path) -> None:
    """Lossless concat via the concat demuxer. No re-encode."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = edit_dir / "_concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in segment_paths))

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"concat → {out_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    concat_list.unlink(missing_ok=True)


# -------- Master SRT (Rule 5) ------------------------------------------------


PUNCT_BREAK = set(".,!?;:")


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _srt_parse(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def scale_srt(src: Path, dst: Path, speed: float) -> None:
    """Reescala un SRT al timeline acelerado (tiempos / speed).

    El SRT que se QUEMA va en el timeline del composite — se acelera junto al
    video, así que queda sincronizado solo. Este archivo aparte es para uso
    externo (subir subtítulos a YouTube), donde el timeline es el del entregable.
    """
    out: list[str] = []
    for line in src.read_text().splitlines():
        m = re.match(r"^(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)\s*$", line)
        if m:
            out.append(f"{_srt_timestamp(_srt_parse(m.group(1)) / speed)} --> "
                       f"{_srt_timestamp(_srt_parse(m.group(2)) / speed)}")
        else:
            out.append(line)
    dst.write_text("\n".join(out))
    print(f"SRT reescalado a {speed:g}x → {dst.name} (para subir aparte)")


def _words_in_range(transcript: dict, t_start: float, t_end: float) -> list[dict]:
    out: list[dict] = []
    for w in transcript.get("words", []):
        if w.get("type") != "word":
            continue
        ws = w.get("start")
        we = w.get("end")
        if ws is None or we is None:
            continue
        if we <= t_start or ws >= t_end:
            continue
        out.append(w)
    return out


def build_master_srt(edl: dict, edit_dir: Path, out_path: Path) -> None:
    """Build an output-timeline SRT from per-source transcripts.

    - 2-word chunks (break on any punctuation in between)
    - UPPERCASE text
    - Output times computed as word.start - segment_start + segment_offset
    """
    transcripts_dir = edit_dir / "transcripts"
    sources = edl["sources"]

    entries: list[tuple[float, float, str]] = []
    seg_offset = 0.0

    for r in edl["ranges"]:
        src_name = r["source"]
        seg_start = float(r["start"])
        seg_end = float(r["end"])
        seg_duration = seg_end - seg_start

        tr_path = transcripts_dir / f"{src_name}.json"
        if not tr_path.exists():
            print(f"  no transcript for {src_name}, skipping captions for this segment")
            seg_offset += seg_duration
            continue

        transcript = json.loads(tr_path.read_text())
        words_in_seg = _words_in_range(transcript, seg_start, seg_end)

        # Group into 2-word chunks, break on punctuation
        chunks: list[list[dict]] = []
        current: list[dict] = []
        for w in words_in_seg:
            text = (w.get("text") or "").strip()
            if not text:
                continue
            current.append(w)
            # Break if the current text ends in punctuation or we hit 2 words
            ends_in_punct = bool(text) and text[-1] in PUNCT_BREAK
            if len(current) >= 2 or ends_in_punct:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        for chunk in chunks:
            local_start = max(seg_start, chunk[0].get("start", seg_start))
            local_end = min(seg_end, chunk[-1].get("end", seg_end))
            out_start = max(0.0, local_start - seg_start) + seg_offset
            out_end = max(0.0, local_end - seg_start) + seg_offset
            if out_end <= out_start:
                out_end = out_start + 0.4
            text = " ".join((w.get("text") or "").strip() for w in chunk)
            text = re.sub(r"\s+", " ", text).strip()
            # Strip trailing punctuation for cleaner uppercase look
            text = text.rstrip(",;:")
            text = text.upper()
            entries.append((out_start, out_end, text))

        seg_offset += seg_duration

    # Sort and write as SRT
    entries.sort(key=lambda e: e[0])
    lines: list[str] = []
    for i, (a, b, t) in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(a)} --> {_srt_timestamp(b)}")
        lines.append(t)
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"master SRT → {out_path.name} ({len(entries)} cues)")


# -------- Loudness normalization (social-ready audio) -----------------------


# Social-media standard: -14 LUFS integrated, -1 dBTP peak, LRA 11 LU.
# Matches YouTube / Instagram / TikTok / X / LinkedIn normalization targets.
LOUDNORM_I = -14.0
LOUDNORM_TP = -1.0
LOUDNORM_LRA = 11.0

# Audio del entregable final. Los clips intermedios se guardan en PCM (lossless)
# para que solo exista UN encode AAC en todo el pipeline (el de aquí). Encodear
# AAC dos veces a 192k (clips + loudnorm) producía ruido de cuantización audible
# en agudos correlacionado con el habla ("estática al hablar"). 256k en una sola
# generación deja el residuo ~45 dB abajo (inaudible). Ver CLAUDE.md.
FINAL_AAC_BITRATE = "256k"


def measure_loudness(video_path: Path) -> dict[str, str] | None:
    """Run ffmpeg loudnorm first pass and parse the JSON measurement.

    Returns a dict with measured_i, measured_tp, measured_lra, measured_thresh,
    target_offset, or None if measurement failed.
    """
    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:print_format=json"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(video_path),
        "-af", filter_str,
        "-vn", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # loudnorm prints the JSON to stderr at the end of the run
    stderr = proc.stderr

    # Find the JSON block — loudnorm output contains a `{ ... }` block
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(stderr[start : end + 1])
    except json.JSONDecodeError:
        return None
    needed = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
    if not needed.issubset(data.keys()):
        return None
    return data


def apply_loudnorm_two_pass(
    input_path: Path,
    output_path: Path,
    preview: bool = False,
) -> bool:
    """Run two-pass loudnorm on input_path, write normalized copy to output_path.

    Returns True on success, False if measurement failed (caller should fall
    back to copying the input unchanged).

    In preview mode, skips the measurement pass and uses a one-pass approximation
    for speed. Final mode always does the proper two-pass.
    """
    if preview:
        # One-pass approximation — faster, slightly less accurate.
        filter_str = f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(input_path),
            "-c:v", "copy",
            "-af", filter_str,
            "-c:a", "aac", "-b:a", FINAL_AAC_BITRATE, "-ar", "48000",
            "-movflags", "+faststart",
            str(output_path),
        ]
        print(f"  loudnorm (1-pass preview) → {output_path.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True

    # Full two-pass
    print(f"  loudnorm pass 1: measuring {input_path.name}")
    measurement = measure_loudness(input_path)
    if measurement is None:
        print("  loudnorm measurement failed — falling back to 1-pass")
        return apply_loudnorm_two_pass(input_path, output_path, preview=True)

    print(f"    measured: I={measurement['input_i']} LUFS  "
          f"TP={measurement['input_tp']}  LRA={measurement['input_lra']}")

    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        f":measured_I={measurement['input_i']}"
        f":measured_TP={measurement['input_tp']}"
        f":measured_LRA={measurement['input_lra']}"
        f":measured_thresh={measurement['input_thresh']}"
        f":offset={measurement['target_offset']}"
        f":linear=true"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(input_path),
        "-c:v", "copy",
        "-af", filter_str,
        "-c:a", "aac", "-b:a", FINAL_AAC_BITRATE, "-ar", "48000",
        "-movflags", "+faststart",
        str(output_path),
    ]
    print(f"  loudnorm pass 2: normalizing → {output_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return True


# -------- Final compositing (Rule 1 + Rule 4) -------------------------------


def build_final_composite(
    base_path: Path,
    overlays: list[dict],
    subtitles_path: Path | None,
    out_path: Path,
    edit_dir: Path,
    encode_audio: bool = False,
    nvenc: bool = False,
    speed: float = 1.0,
) -> None:
    """Final pass: base → overlays (PTS-shifted) → subtitles LAST → speed → out.

    If there are no overlays and no subtitles, just copy base to out.

    El audio de base.mp4 viene en PCM (lossless). Si esta salida es terminal
    (encode_audio=True, p.ej. --no-loudnorm) hay que codear AAC aquí, que es el
    único encode lossy. Si es intermedia (antes de loudnorm) se copia el PCM
    para que loudnorm haga el único encode AAC. Ver FINAL_AAC_BITRATE.

    `speed` (aceleración global) se aplica al FINAL de la cadena, después de
    overlays y subtítulos: así todo se acelera coherentemente y no hay que
    reescalar ningún timestamp. Va antes del loudnorm, que mide el audio ya
    acelerado.
    """
    has_overlays = bool(overlays)
    has_subs = subtitles_path is not None and subtitles_path.exists()
    apply_speed = abs(speed - 1.0) > 1e-6

    # Con speed el audio pasa por atempo y deja de poder copiarse; se reescribe
    # en PCM (lossless) para no gastar la única generación AAC del pipeline.
    if encode_audio:
        audio_args = ["-c:a", "aac", "-b:a", FINAL_AAC_BITRATE, "-ar", "48000"]
    elif apply_speed:
        audio_args = ["-c:a", "pcm_s16le"]
    else:
        audio_args = ["-c:a", "copy"]

    # Composite re-encodea video. x264 fast/crf18 por default; H.264 NVENC con --nvenc.
    vcodec = (
        ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq",
         "-rc", "vbr", "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p"]
        if nvenc else
        ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
    )

    if not has_overlays and not has_subs:
        if apply_speed:
            # Nada que componer pero sí que acelerar → re-encode mínimo
            run(["ffmpeg", "-y", "-i", str(base_path),
                 "-filter_complex",
                 f"[0:v]setpts=PTS/{speed:.6f}[outv];[0:a]atempo={speed:.6f}[outa]",
                 "-map", "[outv]", "-map", "[outa]", *vcodec, *audio_args,
                 "-movflags", "+faststart", str(out_path)], quiet=True)
        else:
            # Nada que componer — copiar video y (re)encodear audio según destino
            run(["ffmpeg", "-y", "-i", str(base_path), "-c:v", "copy", *audio_args, str(out_path)], quiet=True)
        return

    inputs: list[str] = ["-i", str(base_path)]
    for ov in overlays:
        ov_path = resolve_path(ov["file"], edit_dir)
        inputs += ["-i", str(ov_path)]

    filter_parts: list[str] = []
    # PTS-shift every overlay so its frame 0 lands at start_in_output.
    # Three modes:
    #   colorkey present → green-screen key then composite
    #   .webm (no colorkey) → VP9 dual-stream alpha: stream:0=RGB, stream:1=alpha mask
    #   other (MOV/MP4, no colorkey) → native yuva420p alpha channel
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        ck = ov.get("colorkey")
        ov_path = resolve_path(ov["file"], edit_dir)
        is_webm = ov_path.suffix.lower() == ".webm"
        if ck:
            filter_parts.append(f"[{idx}:v]colorkey={ck},format=yuva420p,setpts=PTS-STARTPTS+{t}/TB[a{idx}]")
        elif is_webm:
            # VP9 alpha stores RGB and alpha as two separate video streams in the WebM container.
            # stream:0 = color, stream:1 = alpha (grayscale). Merge them before overlaying.
            filter_parts.append(f"[{idx}:v:0]setpts=PTS-STARTPTS+{t}/TB[ov_rgb{idx}]")
            filter_parts.append(f"[{idx}:v:1]setpts=PTS-STARTPTS+{t}/TB[ov_a{idx}]")
            filter_parts.append(f"[ov_rgb{idx}][ov_a{idx}]alphamerge[a{idx}]")
        else:
            filter_parts.append(f"[{idx}:v]format=yuva420p,setpts=PTS-STARTPTS+{t}/TB[a{idx}]")

    # Chain overlays on top of base
    current = "[0:v]"
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        dur = float(ov["duration"])
        end = t + dur
        next_label = f"[v{idx}]"
        filter_parts.append(
            f"{current}[a{idx}]overlay=enable='between(t,{t:.3f},{end:.3f})'{next_label}"
        )
        current = next_label

    # Subtitles LAST — Rule 1
    if has_subs:
        subs_abs = (str(subtitles_path.resolve())
                    .replace("\\", "/")
                    .replace(":", r"\:")
                    .replace(" ", r"\ ")
                    .replace("'", r"\'"))
        filter_parts.append(
            f"{current}subtitles='{subs_abs}':force_style='{SUB_FORCE_STYLE}'[outv]"
        )
        out_label = "[outv]"
    else:
        # Rename the last overlay output to [outv] for consistency
        if has_overlays:
            filter_parts.append(f"{current}null[outv]")
            out_label = "[outv]"
        else:
            out_label = "[0:v]"

    # Speed al final de todo: los subtítulos ya quemados se aceleran junto al
    # video, así que siguen sincronizados sin tocar el SRT.
    amap = "0:a"
    if apply_speed:
        filter_parts.append(f"{out_label}setpts=PTS/{speed:.6f}[outv_s]")
        out_label = "[outv_s]"
        filter_parts.append(f"[0:a]atempo={speed:.6f}[outa]")
        amap = "[outa]"

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", out_label,
        "-map", amap,
        *vcodec,
        *audio_args,
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"compositing → {out_path.name}")
    print(f"  overlays: {len(overlays)}, subtitles: {'yes' if has_subs else 'no'}"
          + (f", speed: {speed:g}x" if apply_speed else ""))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


# -------- Main ---------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a video from an EDL")
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output video path")
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: 1080p evaluable para QC. Usa H.264 NVENC si hay GPU "
             "(si no, x264 medium CRF 22).",
    )
    ap.add_argument(
        "--draft",
        action="store_true",
        help="Draft mode: 720p, ultrafast, CRF 28 — cut-point verification only.",
    )
    ap.add_argument(
        "--build-subtitles",
        action="store_true",
        help="Build master.srt from transcripts + EDL offsets before compositing",
    )
    ap.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Skip subtitles even if the EDL references one",
    )
    ap.add_argument(
        "--no-loudnorm",
        action="store_true",
        help="Skip audio loudness normalization. Default is on (-14 LUFS, -1 dBTP, LRA 11).",
    )
    ap.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Output framerate. Default 24. Use 60 to preserve 60fps source.",
    )
    ap.add_argument(
        "--high-quality",
        action="store_true",
        help="Final-render high quality (preset slow, CRF 17). Ignored if --preview/--draft.",
    )
    ap.add_argument(
        "--nvenc",
        action="store_true",
        help="Encode con GPU NVIDIA (H.264 NVENC) en vez de x264. ~2.5x más rápido para "
             "ITERAR; a bitrate generoso es transparente para YouTube. Implícito en "
             "--preview (usa --x264 para desactivarlo). Para el entregable final de "
             "máxima calidad-por-bit usa x264 (sin este flag) + --high-quality.",
    )
    ap.add_argument(
        "--x264",
        action="store_true",
        help="Fuerza encode por CPU aunque haya GPU. Desactiva el NVENC implícito de "
             "--preview.",
    )
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Aceleración global del entregable (video setpts + audio atempo). "
             "Default 1.0 (sin cambio). El canal usa 1.08. Se aplica al final, "
             "después de overlays y subtítulos, así que nada se desincroniza. "
             "OJO: en el flujo con stitch (liquid glass por tramos) el speed va "
             "en el stitch, no aquí.",
    )
    args = ap.parse_args()

    if args.nvenc and args.x264:
        sys.exit("--nvenc y --x264 son mutuamente excluyentes.")
    if not (0.5 <= args.speed <= 2.0):
        sys.exit("--speed fuera de rango: atempo admite 0.5-2.0 en una pasada.")
    if args.speed > 1.15:
        print(f"  aviso: speed {args.speed:g}x es alto; por encima de ~1.10 la "
              f"aceleración se nota en tramos densos.")
    if args.nvenc and not nvenc_available():
        sys.exit("--nvenc pedido pero este ffmpeg no expone h264_nvenc. "
                 "Verifica GPU/drivers NVIDIA y que ffmpeg esté compilado con NVENC "
                 "(`ffmpeg -hide_banner -encoders | findstr nvenc`).")

    # --preview implica NVENC: el preview es para ITERAR, y H.264 por GPU cuesta
    # ~2.5x menos sin perder resolución (medido: 5m43s vs 14m21s en 279 clips).
    # Fallback silencioso a x264 si la máquina no tiene GPU.
    if args.preview and not args.nvenc and not args.x264 and nvenc_available():
        args.nvenc = True
        print("  preview: usando H.264 NVENC (GPU). --x264 para forzar CPU.")

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    edl = json.loads(edl_path.read_text())
    edit_dir = edl_path.parent
    out_path = args.output.resolve()

    # 1. Extract per-segment (auto-grade per range if EDL grade is "auto")
    segment_paths = extract_all_segments(
        edl, edit_dir, preview=args.preview, draft=args.draft,
        fps=args.fps, high_quality=args.high_quality, nvenc=args.nvenc,
    )

    # 2. Concat → base
    if args.draft:
        base_name = "base_draft.mp4"
    elif args.preview:
        base_name = "base_preview.mp4"
    else:
        base_name = "base.mp4"
    base_path = edit_dir / base_name
    concat_segments(segment_paths, base_path, edit_dir)

    # 3. Subtitles: build if requested, resolve final path
    subs_path: Path | None = None
    if not args.no_subtitles:
        if args.build_subtitles:
            subs_path = edit_dir / "master.srt"
            build_master_srt(edl, edit_dir, subs_path)
        elif edl.get("subtitles"):
            subs_path = resolve_path(edl["subtitles"], edit_dir)
            if not subs_path.exists():
                print(f"warning: subtitles path in EDL does not exist: {subs_path}")
                subs_path = None
    if subs_path and abs(args.speed - 1.0) > 1e-6:
        scale_srt(subs_path, subs_path.with_name(f"{subs_path.stem}_speed.srt"), args.speed)

    # 4. Composite (overlays + subtitles LAST) → intermediate (pre-loudnorm) path
    overlays = edl.get("overlays") or []
    if args.no_loudnorm:
        # Composite directly to final output → salida terminal, encodear AAC aquí
        # (base.mp4 trae PCM; sin loudnorm este es el único encode lossy).
        build_final_composite(base_path, overlays, subs_path, out_path, edit_dir,
                              encode_audio=True, nvenc=args.nvenc, speed=args.speed)
    else:
        # Composite to a temp file, then run loudnorm → final output
        tmp_composite = out_path.with_suffix(".prenorm.mp4")
        build_final_composite(base_path, overlays, subs_path, tmp_composite, edit_dir,
                              nvenc=args.nvenc, speed=args.speed)
        print("loudness normalization → social-ready (-14 LUFS / -1 dBTP / LRA 11)")
        apply_loudnorm_two_pass(tmp_composite, out_path, preview=args.draft)
        tmp_composite.unlink(missing_ok=True)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\ndone: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
