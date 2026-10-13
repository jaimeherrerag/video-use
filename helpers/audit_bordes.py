"""Auditor de bordes: que se dice alrededor de cada corte, medido con silencios reales.

Por que existe (short de REA, 2026-10-12): los cortes se ponen con los timestamps
del ASR, y WhisperX los corre sin avisar. Un numero ("150.000" = "ciento cincuenta
mil", 1.6 s) se alinea en 0.3 s y empuja todo lo que sigue; una retoma se duplica
como palabras fantasma. Tres rondas de correcciones cortaron "Anything",
"estrellas" y "en GitHub" porque cada boundary se puso segun palabras que no
estaban donde el transcript decia.

Lo que si funciono: partir la zona por silencios REALES (energia) y transcribir
cada bloque de voz por separado. Este helper hace eso alrededor de cada corte:

  - ERROR corte-en-bloque: el corte cae a mitad de un bloque de voz.
  - WARN  tras-el-corte / antes-del-corte: a <0.6 s del corte hay un bloque con
          palabras que no entro al video (frase incompleta? o descarte legitimo).
  - WARN  desalineado: el bloque dice una cosa y el transcript pone ahi otra
          (o nada). Los cortes de esa zona no son confiables.
  - INFO  numero: el range tiene un numero; sus tiempos son poco confiables y se
          audita completo, no solo en los bordes.

Las alucinaciones conocidas de Whisper sobre ruido ("¡Suscribete al canal!",
"Gracias por ver") se marcan como ruido, nunca como palabras.

Uso (en .venv-asr, desde video-use/):
    .venv-asr/Scripts/python.exe helpers/audit_bordes.py --edit-dir "<edit>"
    .venv-asr/Scripts/python.exe helpers/audit_bordes.py --edit-dir "<edit>" --solo 4,5
Salida: <edit>/verify/bordes_report.md y .json. Exit 1 si hay ERROR.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_edl import (HABLA_DB, RE_MARCA, VENTANA_MS, cargar, db_pico,  # noqa: E402
                        es_palabra, fmt_t, mapa_energia, resolve_path, txt)

BLOQUE_DB = -42.0       # umbral de voz para partir en bloques (el -42 dB/0.06 s que resolvio el caso REA)
SILENCIO_MIN = 0.06     # silencio minimo que separa dos bloques
BLOQUE_MIN = 0.12       # bloques mas cortos = clicks o respiraciones
GAP_FRASE = 0.60        # un bloque a menos de esto del corte puede ser la misma frase
VECINDAD_ANTES = 2.0    # cuanto audio se audita del lado que se queda
VECINDAD_DESPUES = 1.5  # y del lado que se corta
DESAL_MIN = 0.25        # bloques mas cortos no sirven para juzgar alineacion (Whisper adivina)
VACIO_MIN = 0.35        # "voz sin palabras en el transcript" solo cuenta en bloques de este largo
PERDIDA_MIN = 0.10      # audio perdido por un corte: menos es el arranque del sonido, no se oye
PALABRAS_POR_S = 6.0    # mas palabras que esto por segundo de bloque = alucinacion de Whisper
BLOQUE_MAX = 8.0        # bloques mas largos se recortan (transcribir minutos no aporta)
PAD = 0.08              # margen de audio alrededor de cada bloque (cae en silencio)

RE_ALUCINACION = re.compile(
    r"suscr[ií]b|gracias por ver|subt[ií]tul|amara\.org|nos vemos en el pr[oó]ximo|"
    r"dale like|activa la campanita|m[aá]s videos", re.I)
NUM_PALABRAS = {"cero", "uno", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
                "diez", "once", "doce", "quince", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta",
                "setenta", "ochenta", "noventa", "cien", "ciento", "doscientos", "trescientos", "quinientos",
                "mil", "millon", "millones", "y"}


def eco_prompt(texto: str) -> bool:
    """Whisper sobre casi-silencio devuelve su initial_prompt ('VisiumAI, Claude Code, N8n...')."""
    from transcribe_local import DEFAULT_PROMPT
    terminos = {norm(x) for x in re.split(r"[,.;]", DEFAULT_PROMPT) if len(x.strip()) > 3}
    t = norm(texto)
    return sum(1 for x in terminos if x and x in t) >= 3


def pico_db(audio: np.ndarray, t0: float, t1: float) -> float:
    x = audio[int(max(0.0, t0) * 16000): int(t1 * 16000)]
    if len(x) == 0:
        return -99.0
    n = int(FRAME * 16000)
    k = max(1, len(x) // n)
    rms = np.sqrt((x[: k * n].reshape(k, n) ** 2).mean(axis=1)) if len(x) >= n else np.array([np.sqrt((x ** 2).mean())])
    return float(20.0 * np.log10(max(float(rms.max()), 1e-10)))


def norm(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[.,;:¿?¡!…\"'()]", "", t)
    t = (t.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o")
          .replace("ú", "u").replace("ü", "u"))
    return t


def tokens(texto: str) -> list[str]:
    """Tokens normalizados; cualquier numero (en cifra o en palabras) se vuelve <num>."""
    out = []
    for t in texto.split():
        n = norm(t)
        if not n:
            continue
        if re.search(r"\d", n) or (n in NUM_PALABRAS and n != "y"):
            if not out or out[-1] != "<num>":
                out.append("<num>")
        else:
            out.append(n)
    return out


FRAME = 0.010           # resolucion para partir bloques (el mapa de 50 ms funde pausas de 90 ms)


def bloques_voz(audio: np.ndarray, t0: float, t1: float) -> list[tuple[float, float]]:
    """Bloques de voz en [t0,t1], separados por silencios >= SILENCIO_MIN a BLOQUE_DB.

    Se mide sobre el audio en ventanas de 10 ms (no con el mapa de 50 ms de
    verify_edl: ahi la pausa de 90 ms entre "mil" y "estrellas" desaparecia).
    Se buscan en una ventana mas ancha y se devuelven completos: un bloque que
    empieza antes de t0 no se recorta, porque transcribir media palabra es
    justo el error que se quiere evitar.
    """
    a0, a1 = max(0.0, t0 - BLOQUE_MAX), min(len(audio) / 16000.0, t1 + BLOQUE_MAX)
    n = int(FRAME * 16000)
    x = audio[int(a0 * 16000): int(a1 * 16000)]
    k = len(x) // n
    if k == 0:
        return []
    rms = np.sqrt((x[: k * n].reshape(k, n) ** 2).mean(axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    mask = db > BLOQUE_DB
    runs, ini_ = [], None
    for i, v in enumerate(mask):
        if v and ini_ is None:
            ini_ = i
        elif not v and ini_ is not None:
            runs.append([a0 + ini_ * FRAME, a0 + i * FRAME, ini_, i])
            ini_ = None
    if ini_ is not None:
        runs.append([a0 + ini_ * FRAME, a0 + k * FRAME, ini_, k])
    fus = []
    for r in runs:
        if fus and r[0] - fus[-1][1] < SILENCIO_MIN - 1e-6:
            fus[-1][1], fus[-1][3] = r[1], r[3]
        else:
            fus.append(r)
    out = []
    for s, e, i0, i1 in fus:
        if e < t0 or s > t1 or e - s < BLOQUE_MIN:
            continue
        if float(db[i0:i1].max()) <= HABLA_DB:      # respiracion o ruido de fondo
            continue
        out.append((round(s, 3), round(e, 3)))
    return out


class Transcriptor:
    """faster-whisper large-v3 cargado una vez; transcribe bloques cortos sin VAD."""

    def __init__(self, prompt: str):
        import torch  # primero: registra torch\\lib en el DLL search path (cudnn)
        from faster_whisper import WhisperModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = WhisperModel("large-v3", device=dev, compute_type="float16" if dev == "cuda" else "int8")
        self.prompt = prompt
        self.cache: dict[tuple, str] = {}

    def __call__(self, audio: np.ndarray, s: float, e: float) -> str:
        clave = (round(s, 2), round(e, 2))
        if clave not in self.cache:
            a = audio[int(max(0.0, s - PAD) * 16000): int(min(e, s + BLOQUE_MAX) * 16000 + PAD * 16000)]
            segs, _ = self.m.transcribe(a, language="es", initial_prompt=self.prompt, vad_filter=False,
                                        beam_size=5, condition_on_previous_text=False)
            self.cache[clave] = " ".join(x.text.strip() for x in segs).strip()
        return self.cache[clave]


def cargar_audio(src: Path) -> np.ndarray:
    import whisperx
    return whisperx.load_audio(str(src))


def auditar(edl_path: Path, solo: set[int] | None):
    edl, edit_dir, transcripts = cargar(edl_path)
    ranges = edl.get("ranges", [])
    outs, cum = [], 0.0
    for r in ranges:
        outs.append(cum)
        cum += float(r["end"]) - float(r["start"])

    from transcribe_local import DEFAULT_PROMPT
    tr = Transcriptor(DEFAULT_PROMPT)
    energia, audios = {}, {}
    hallazgos = []

    def add(sev, tipo, i, borde, out_t, src_t, msg, texto="", contexto=""):
        hallazgos.append(dict(sev=sev, tipo=tipo, range=i, borde=borde, out=round(out_t, 2),
                              src=round(src_t, 2), msg=msg, texto=texto, contexto=contexto))

    def en_rango(key, t0, t1):
        """True si [t0,t1] ya entra al video en algun range de ese source."""
        return any(r.get("source") == key and float(r["start"]) <= t0 + 0.05 and float(r["end"]) >= t1 - 0.05
                   for r in ranges)

    def words_en(key, t0, t1):
        return [w for w in transcripts.get(key, []) if es_palabra(w) and t0 - 0.02 <= float(w["start"]) < t1]

    def words_solapan(key, t0, t1):
        return [w for w in transcripts.get(key, []) if es_palabra(w) and float(w["end"]) > t0 + 0.02 and float(w["start"]) < t1 - 0.02]

    for i, r in enumerate(ranges):
        if solo and i not in solo:
            continue
        key = r.get("source")
        src = resolve_path(edl["sources"][key], edit_dir)
        if key not in energia:
            energia[key] = mapa_energia(src, edit_dir / "verify")
            audios[key] = cargar_audio(src)
        db, audio = energia[key], audios[key]
        if db is None:
            continue
        s, e = float(r["start"]), float(r["end"])
        con_numero = any(re.search(r"\d", txt(w)) for w in words_en(key, s, e))
        if con_numero:
            add("INFO", "numero", i, "", outs[i], s,
                "el range tiene un numero: sus timestamps son poco confiables; se audita completo")

        # zonas a auditar: la cola (end) y la cabeza (start); con numero, el range entero
        zonas = [("end", e - VECINDAD_ANTES, e + VECINDAD_DESPUES), ("start", s - VECINDAD_DESPUES, s + VECINDAD_ANTES)]
        if con_numero:
            zonas.append(("todo", s, e))
        vistos = set()
        for borde, z0, z1 in zonas:
            for b0, b1 in bloques_voz(audio, z0, z1):
                if (b0, b1, borde) in vistos:
                    continue
                vistos.add((b0, b1, borde))
                texto = tr(audio, b0, b1)
                ruido = (not tokens(texto) or bool(RE_ALUCINACION.search(texto)) or eco_prompt(texto)
                         or len(texto.split()) > max(2.0, (b1 - b0) * PALABRAS_POR_S))
                out_b = outs[i] + (b0 - s)

                if borde == "end":
                    if (b0 < e - 0.03 and b1 - e >= PERDIDA_MIN and not ruido
                            and pico_db(audio, e, min(b1, e + 0.5)) > HABLA_DB):
                        add("ERROR", "corte-en-bloque", i, "end", outs[i] + (e - s), e,
                            f"el corte cae a mitad de un bloque de voz ({b0:.2f}-{b1:.2f}): "
                            f"se pierden {b1 - e:.2f}s", texto)
                    elif b0 >= e - 0.03 and b0 - e <= GAP_FRASE and not ruido and not en_rango(key, b0, b1):
                        sev = "INFO" if RE_MARCA.search(texto) else "WARN"
                        add(sev, "tras-el-corte", i, "end", outs[i] + (e - s), b0,
                            f"a {b0 - e:.2f}s del corte se dice algo que no entra al video",
                            texto, " ".join(txt(w) for w in words_en(key, e - 2.5, e)))
                elif borde == "start":
                    if (b1 > s + 0.03 and s - b0 >= PERDIDA_MIN and not ruido
                            and pico_db(audio, max(b0, s - 0.5), s) > HABLA_DB):
                        add("ERROR", "corte-en-bloque", i, "start", outs[i], s,
                            f"el corte arranca a mitad de un bloque de voz ({b0:.2f}-{b1:.2f}): "
                            f"se pierden {s - b0:.2f}s", texto)
                    elif b1 <= s + 0.03 and s - b1 <= GAP_FRASE and not ruido and not en_rango(key, b0, b1):
                        sev = "INFO" if RE_MARCA.search(texto) else "WARN"
                        add(sev, "antes-del-corte", i, "start", outs[i], b0,
                            f"a {s - b1:.2f}s antes del corte se dice algo que no entra al video",
                            texto, " ".join(txt(w) for w in words_en(key, s, s + 2.5)))

                # alineacion: lo que el bloque dice vs lo que el transcript pone ahi (solo lo que se queda)
                if not ruido and (borde == "todo" or en_rango(key, b0, b1)):
                    tb = set(tokens(texto))
                    tt = set(tokens(" ".join(txt(w) for w in words_solapan(key, b0, b1))))
                    cerca = set(tokens(" ".join(txt(w) for w in words_en(key, b0 - 3.0, b0) + words_en(key, b1, b1 + 3.0))))
                    corrido = len((tb - tt) & cerca) >= 2      # al menos 2 palabras del bloque estan corridas a otro lado
                    vacio = not tt and len(tb) >= 1 and b1 - b0 >= VACIO_MIN   # voz con palabras y el transcript no tiene nada
                    if b1 - b0 >= DESAL_MIN and tb and (vacio or corrido) and len(tb & tt) / len(tb | tt) < 0.34:
                        clave = ("desal", round(b0, 2))
                        if clave not in vistos:
                            vistos.add(clave)
                            add("WARN", "desalineado", i, borde, out_b, b0,
                                f"bloque {b0:.2f}-{b1:.2f}: el audio dice otra cosa que el transcript; "
                                "los cortes cercanos no son confiables", texto,
                                "transcript: " + (" ".join(txt(w) for w in words_solapan(key, b0, b1)) or "(nada)"))
    orden = {"ERROR": 0, "WARN": 1, "INFO": 2}
    hallazgos.sort(key=lambda h: (orden[h["sev"]], h["out"]))
    return hallazgos, cum


def main() -> int:
    ap = argparse.ArgumentParser(description="Auditor de bordes con silencios reales")
    ap.add_argument("--edit-dir", type=Path, required=True)
    ap.add_argument("--edl", type=Path, default=None, help="default: <edit>/edl.json")
    ap.add_argument("--solo", type=str, default="", help="indices de range, separados por coma")
    args = ap.parse_args()
    edl_path = (args.edl or args.edit_dir / "edl.json").resolve()
    solo = {int(x) for x in args.solo.split(",") if x.strip()} or None

    t0 = time.time()
    hallazgos, dur = auditar(edl_path, solo)
    vdir = edl_path.parent / "verify"
    vdir.mkdir(exist_ok=True)
    (vdir / "bordes_report.json").write_text(json.dumps(hallazgos, ensure_ascii=False, indent=2), encoding="utf-8")
    lineas = [f"# Auditoría de bordes — {edl_path.parent.parent.name}", "",
              f"Salida {fmt_t(dur)} · {len(hallazgos)} hallazgos · {time.time() - t0:.0f} s", ""]
    for h in hallazgos:
        lineas.append(f"- **{h['sev']}** `{h['tipo']}` r{h['range']} {h['borde']} · out {fmt_t(h['out'])} · "
                      f"src {h['src']:.2f} — {h['msg']}")
        if h["texto"]:
            lineas.append(f"  - audio: \"{h['texto']}\"")
        if h["contexto"]:
            lineas.append(f"  - contexto: {h['contexto']}")
    (vdir / "bordes_report.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    for h in hallazgos:
        print(f"  {h['sev']:5} {h['tipo']:15} r{h['range']:<3} {h['borde']:5} out {fmt_t(h['out'])} "
              f"src {h['src']:8.2f} | {h['texto'][:70]}")
    n_err = sum(h["sev"] == "ERROR" for h in hallazgos)
    print(f"\n{n_err} ERROR | {sum(h['sev'] == 'WARN' for h in hallazgos)} WARN | "
          f"{sum(h['sev'] == 'INFO' for h in hallazgos)} INFO  ({time.time() - t0:.0f} s) -> {vdir / 'bordes_report.md'}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
