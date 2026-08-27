#!/usr/bin/env python3
"""verify_edl.py - checklist automatico post-EDL (Presets 3, 4, 5, 6 del CLAUDE.md).

Corre TODAS las verificaciones documentadas antes de renderizar un preview, para
que los defectos que el QC de senal no ve (pausas, repeticiones, habla omitida por
el ASR) se detecten sin depender del criterio de cada sesion.

Uso:
    uv run python helpers/verify_edl.py <edl.json> [--json reporte.json] [--strict]

Diseno: un SOLO pase de energia por source (cacheado en edit/verify/), no un
silencedetect por range - con 300+ ranges eso tardaba minutos. El mapa de energia
a -38 dB es la referencia (silencedetect a -32 dB corta conectivas suaves como
"y"/"que", que miden -33..-38 dB; ver el gotcha de la regresion 2026-08-04).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Umbrales - todos vienen de reglas ya documentadas en CLAUDE.md
# ---------------------------------------------------------------------------
PAUSA_INTRA_MAX = 0.50   # "cortar TODA pausa >=0.5s" (preferencia de ritmo 2026-07-13)
GAP_INTER_MAX = 0.25     # Preset 5: gap inter-range maximo 250 ms
HEAD_MAX = 0.50          # head grande de un solo lado (la juntura se mide aparte)
TAIL_MAX = 0.50          # idem tail; tipicamente el ASR estirando la ultima word
DUR_MIN = 1.0            # ranges <1s = posible fragmento truncado
DUR_MAX = 25.0           # ranges >25s = probable pausa interna no detectada
WORD_ESTIRADA = 0.50     # piso: word corta >0.5s = estirada (CLAUDE.md; normal ~0.2s)
CHARS_POR_S = 14.0       # ritmo del espanol, para escalar el umbral con el largo
HUECO_VOZ_MIN = 0.35     # hueco sin words dentro de voz: >=0.35s es sospechoso
VOZ_DB = -38.0           # umbral del mapa de energia (NO -32: corta conectivas)
HABLA_DB = -30.0         # hueco con RMS mayor a esto = habla perdida, no respiracion
VENTANA_MS = 50          # RMS por ventanas de 50 ms
BLIP_MAX = 0.10          # run de voz mas corto que esto = click, no habla
MARGEN_CLAMP = 2.5       # cuanto puede exceder una word su duracion esperada
                         # antes de considerarla estirada (2.5x deja pasar
                         # URLs y dominios leidos enteros)

# Marcas verbales de corte. "por eso" va aparte: WhisperX transcribe asi un
# "borra eso" rapido, pero tambien es una locucion legitima -> severidad INFO.
RE_MARCA = re.compile(r"\b(cort[ao]\s+eso|c[oó]rtalo|b[oó]rra\w{0,3}\s+eso)\b", re.I)
RE_MARCA_DUDOSA = re.compile(r"\bpor\s+eso\b", re.I)
RE_FILLER = re.compile(r"^(m+h*m+|a+m+|e+h+|a+h+|u+h+)[.,]?$", re.I)

SEV_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def fmt_t(s: float) -> str:
    """Segundos -> MM:SS.mm (floor, no banker's rounding)."""
    m = int(s // 60)
    return f"{m:02d}:{s - m * 60:05.2f}"


def norm_token(t: str) -> str:
    """Minusculas sin acentos ni puntuacion, para comparar repeticiones."""
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ]", "", t)


def resolve_path(maybe: str, base: Path) -> Path:
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def es_palabra(w: dict) -> bool:
    if w.get("type") != "word":
        return False
    if w.get("start") is None or w.get("end") is None:
        return False
    return bool(re.search(r"[0-9A-Za-zÀ-ſ]", w.get("text") or ""))


def txt(w: dict) -> str:
    return (w.get("text") or "").strip()


def dur_esperada(texto: str) -> float:
    """Duracion maxima plausible de una word segun su largo.

    El espanol ronda 14 caracteres/s en habla normal. Un umbral FIJO marca como
    'estirada' cualquier palabra larga pronunciada entera, que es lo normal;
    lo sospechoso es una palabra CORTA que ocupa mucho tiempo (ahi el ASR estiro
    el token sobre silencio o sobre habla que no transcribio).

    Las SIGLAS se deletrean: "CRM" son 3 caracteres pero suenan como "ce-ere-eme"
    (~1s). Sin este ajuste toda sigla parece estirada - y en este canal salen a
    cada rato (CRM, VPS, API, SSH, MCP).
    """
    n = len(texto)
    if n >= 2 and texto.isupper():
        n *= 3
    return max(WORD_ESTIRADA, n / CHARS_POR_S + 0.25)


# ---------------------------------------------------------------------------
# Mapa de energia (un pase por source, cacheado)
# ---------------------------------------------------------------------------
def mapa_energia(src: Path, cache_dir: Path):
    """dB RMS por ventana de VENTANA_MS. Cachea en edit/verify/<stem>.energy.npy."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{src.stem}.energy.npy"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return np.load(cache)

    print(f"  midiendo energia de {src.name} (una vez, luego se cachea)...")
    cmd = ["ffmpeg", "-v", "error", "-i", str(src),
           "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  !! no se pudo leer audio de {src.name}: {e}")
        return None

    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    win = int(16000 * VENTANA_MS / 1000)
    n = len(a) // win
    if n == 0:
        return None
    rms = np.sqrt((a[: n * win].reshape(n, win) ** 2).mean(axis=1))
    db = (20.0 * np.log10(np.maximum(rms, 1e-10))).astype(np.float32)
    np.save(cache, db)
    return db


def runs_voz(db, t0: float, t1: float, umbral: float = VOZ_DB):
    """Tramos con energia > umbral dentro de [t0,t1], en segundos absolutos."""
    if db is None:
        return []
    step = VENTANA_MS / 1000.0
    i0, i1 = max(0, int(t0 / step)), min(len(db), int(t1 / step) + 1)
    if i1 <= i0:
        return []
    mask = db[i0:i1] > umbral
    out, ini = [], None
    for i, v in enumerate(mask):
        if v and ini is None:
            ini = i
        elif not v and ini is not None:
            out.append(((i0 + ini) * step, (i0 + i) * step))
            ini = None
    if ini is not None:
        out.append(((i0 + ini) * step, i1 * step))
    return out


def fusionar_runs(runs, hueco_max: float = 0.15):
    """Une runs de voz separados por micro-silencios.

    Dentro de UNA palabra hay huecos: las oclusivas (/p/, /t/, /k/) y sobre todo
    las siglas deletreadas ("CRM" = ce-ere-eme). Tratarlos como runs distintos
    hace que el clamp de una word se quede con una sola letra y marque el resto
    como habla perdida.
    """
    if not runs:
        return []
    out = [list(runs[0])]
    for s, e in runs[1:]:
        if s - out[-1][1] <= hueco_max:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(r) for r in out]


def db_pico(db, t0: float, t1: float) -> float:
    if db is None:
        return -99.0
    step = VENTANA_MS / 1000.0
    i0, i1 = max(0, int(t0 / step)), min(len(db), int(t1 / step) + 1)
    if i1 <= i0:
        return -99.0
    return float(db[i0:i1].max())


# ---------------------------------------------------------------------------
# Carga del EDL + transcripts
# ---------------------------------------------------------------------------
def cargar(edl_path: Path):
    edl = json.loads(edl_path.read_text(encoding="utf-8"))
    edit_dir = edl_path.parent
    tdir = edit_dir / "transcripts"
    transcripts = {}
    if tdir.is_dir():
        for key in edl.get("sources", {}):
            tp = tdir / f"{key}.json"
            if tp.exists():
                d = json.loads(tp.read_text(encoding="utf-8"))
                ws = [w for w in d.get("words", []) if es_palabra(w)]
                ws.sort(key=lambda w: float(w["start"]))
                transcripts[key] = ws
    return edl, edit_dir, transcripts


def words_de(transcripts, key, a: float, b: float):
    """Words CONTENIDAS en [a,b] - para analisis de contenido (repeticiones, etc)."""
    ws = transcripts.get(key) or []
    return [w for w in ws
            if float(w["start"]) >= a - 0.02 and float(w["end"]) <= b + 0.02]


def words_solapan(transcripts, key, a: float, b: float):
    """Words que SOLAPAN [a,b] - para cobertura de audio.

    Un corte suele caer a mitad de la ultima word (o el ASR la reporta estirada
    mas alla del boundary). Si esas se excluyen, su audio parece "voz sin words"
    y el check de habla-perdida dispara en falso.
    """
    ws = transcripts.get(key) or []
    return [w for w in ws
            if float(w["end"]) > a and float(w["start"]) < b]


def cobertura(w, db=None) -> tuple:
    """Tramo de audio que una word explica de forma plausible.

    NO se usa su ventana declarada tal cual: el fallo principal de WhisperX es
    estirar una word sobre habla que no transcribio, y darla por buena hace que
    la propia word tape el hueco que se quiere detectar. Peor aun, estira hacia
    cualquiera de los dos lados (visto: "simplemente" declarada 744.09-749.19,
    5.10s, cuando el audio real dura 0.40s al final de esa ventana) - asi que
    recortar desde el `start` tampoco sirve.

    Regla del CLAUDE.md: clampear cada word a su run de voz. Si la ventana
    declarada es plausible se toma entera; si esta estirada, se busca dentro de
    ella el run de voz que la explica y el resto queda descubierto (que es
    justamente donde aparecen las marcas "borra eso" y el habla omitida).
    """
    s, e = float(w["start"]), float(w["end"])
    plausible = dur_esperada(txt(w)) * MARGEN_CLAMP
    if e - s <= plausible or db is None:
        return (s, e)
    runs = fusionar_runs(runs_voz(db, s, e))
    if not runs:
        return (s, e)
    # el run mas largo que cabe en lo plausible; si ninguno cabe, el mas largo
    caben = [r for r in runs if r[1] - r[0] <= plausible]
    elegido = max(caben or runs, key=lambda r: r[1] - r[0])
    return elegido


# ---------------------------------------------------------------------------
# Los checks
# ---------------------------------------------------------------------------
def verificar(edl, edit_dir, transcripts, energia):
    flags = []

    def add(sev, check, idx, out_t, msg, src_t=None):
        flags.append(dict(sev=sev, check=check, range=idx, out=out_t,
                          src=src_t, msg=msg))

    ranges = edl.get("ranges", [])
    heads_tails: dict[int, tuple] = {}   # i -> (head, tail, 1a word, ult word, t_ult, t_ini)
    # mapeo output-time (suma teorica; el concat agrega ~20ms/segmento)
    outs, cum = [], 0.0
    for r in ranges:
        outs.append(cum)
        cum += float(r["end"]) - float(r["start"])

    # --- Check 5: sources existen -----------------------------------------
    sources = edl.get("sources", {})
    for i, r in enumerate(ranges):
        if r.get("source") not in sources:
            add("ERROR", "sources", i, outs[i],
                f"source '{r.get('source')}' no esta en edl.sources")
    for key, p in sources.items():
        fp = resolve_path(p, edit_dir)
        if not fp.exists():
            add("ERROR", "sources", -1, 0.0, f"source '{key}' no existe en disco: {fp}")
        if key not in transcripts:
            add("WARN", "sources", -1, 0.0,
                f"source '{key}' sin transcript en transcripts/{key}.json "
                f"(render.py generara SRT vacio para sus ranges)")

    # --- Checks por range --------------------------------------------------
    for i, r in enumerate(ranges):
        key = r.get("source")
        s, e = float(r["start"]), float(r["end"])
        dur = e - s
        out_t = outs[i]
        db = energia.get(key)
        ws = words_de(transcripts, key, s, e)

        # Check 4: duration outliers
        if dur < DUR_MIN:
            add("WARN", "duracion", i, out_t,
                f"range de {dur:.2f}s (<{DUR_MIN}s) - posible fragmento truncado", s)
        elif dur > DUR_MAX:
            add("WARN", "duracion", i, out_t,
                f"range de {dur:.1f}s (>{DUR_MAX}s) - revisar pausa interna", s)

        if not ws:
            if key in transcripts and db is not None and runs_voz(db, s, e):
                add("INFO", "sin-words", i, out_t,
                    "range sin words en el transcript pero CON voz "
                    "(pista desktop / demo?)", s)
            continue

        # Check 1a: pausas internas >= 0.5s (Preset 4).
        # La duracion NO se toma del transcript: entre dos words puede haber una
        # estirada que invente el hueco (o que lo tape). Cuando hay mapa de
        # energia, la pausa real es el silencio medido en ese tramo.
        for a, b in zip(ws, ws[1:]):
            ta, tb = float(a["end"]), float(b["start"])
            gap = tb - ta
            if gap < PAUSA_INTRA_MAX:
                continue
            # El hueco entre dos words ES tiempo muerto para el espectador aunque
            # tenga un ruidito en medio, asi que siempre se reporta. Lo que anade
            # la energia es COMO cortarlo: cuanto silencio contiguo hay, y si lo
            # que interrumpe es voz (entonces no es una pausa que se pueda cortar,
            # es habla que el ASR no transcribio).
            nota = ""
            if db is not None:
                # Los runs <BLIP_MAX no son habla: son clicks de boca/teclado.
                # Sin filtrarlos, un blip de 20ms parte un silencio de 0.9s en
                # dos mitades de <0.5s (gotcha "fusionar silencios separados
                # por blips" del CLAUDE.md).
                voz = [r for r in runs_voz(db, ta, tb) if r[1] - r[0] >= BLIP_MAX]
                bordes = [ta] + [x for r in voz for x in r] + [tb]
                quietos = [(bordes[k], bordes[k + 1])
                           for k in range(0, len(bordes) - 1, 2)]
                mudo = max((q[1] - q[0] for q in quietos), default=0.0)
                voz_total = sum(r[1] - r[0] for r in voz)
                if voz_total >= 0.15:
                    nota = (f" | {voz_total:.2f}s de VOZ dentro del hueco: "
                            f"revisar si es habla no transcrita antes de cortar")
                elif mudo < PAUSA_INTRA_MAX:
                    nota = f" | silencio contiguo max {mudo:.2f}s (cortar con cuidado)"
                else:
                    nota = f" | {mudo:.2f}s de silencio contiguo para cortar"
            add("ERROR", "pausa-intra", i, out_t + (ta - s),
                f"pausa de {gap:.2f}s dentro del range "
                f"(...{txt(a)} | {txt(b)}...){nota}", ta)

        # Check 1b: head / tail silencioso.
        # El silencio de juntura combinado se evalua despues (check "juntura");
        # aqui solo se marca el head/tail grande de un solo lado, que apunta a
        # otra causa: el ASR estirando una word o habla que no transcribio.
        head = float(ws[0]["start"]) - s
        tail = e - float(ws[-1]["end"])
        heads_tails[i] = (head, tail, txt(ws[0]), txt(ws[-1]),
                          float(ws[-1]["end"]), s)
        if head > HEAD_MAX:
            add("WARN", "head", i, out_t,
                f"{head:.2f}s de silencio antes de la 1a palabra ('{txt(ws[0])}')", s)
        if tail > TAIL_MAX:
            ultimo = txt(ws[-1])
            extra = ""
            if re.search(r"\d", ultimo):
                extra = (" - token NUMERICO: su timestamp suele venir truncado, "
                         "usar el fin del run de voz, no el del word")
            add("WARN", "tail", i, out_t + (float(ws[-1]["end"]) - s),
                f"{tail:.2f}s de silencio tras la ultima palabra ('{ultimo}'){extra}",
                float(ws[-1]["end"]))

        # Check 6: words estiradas (tapan habla real).
        # Umbral PROPORCIONAL al largo del texto: "y" de 0.5s esta estirada,
        # "automatizacion" de 0.9s es normal. Con umbral fijo salian 160 flags
        # por video (irrevisable) y casi todos eran palabras largas legitimas.
        for w in ws:
            wd = float(w["end"]) - float(w["start"])
            if wd > dur_esperada(txt(w)):
                sev = "INFO" if db is not None else "WARN"
                add(sev, "word-estirada", i, out_t + (float(w["start"]) - s),
                    f"word '{txt(w)}' dura {wd:.2f}s "
                    f"(esperado <{dur_esperada(txt(w)):.2f}s) - "
                    "puede estar tapando habla no transcrita",
                    float(w["start"]))

        # Check 7: habla sin words (el fallo principal de WhisperX).
        # Cobertura con words que SOLAPAN el range (no solo las contenidas) y
        # recortada a la duracion plausible de cada una - ver cobertura().
        if db is not None:
            cubierto = [cobertura(w, db) for w in
                        words_solapan(transcripts, key, s - 1.0, e + 1.0)]
            for rs, rend in runs_voz(db, s, e):
                libre = [(rs, rend)]
                for cs, ce in cubierto:
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
                    if le - ls < HUECO_VOZ_MIN:
                        continue
                    pico = db_pico(db, ls, le)
                    if pico > HABLA_DB:
                        add("ERROR", "habla-perdida", i, out_t + (ls - s),
                            f"{le - ls:.2f}s de voz SIN words (pico {pico:.0f} dB) - "
                            "re-transcribir el clip con boundaries en silencio", ls)

        # Check 8: marcas verbales y filler que sobrevivieron
        texto = " ".join(txt(w) for w in ws)
        m = RE_MARCA.search(texto)
        if m:
            add("ERROR", "marca-corte", i, out_t,
                f"marca de corte en el output: '{m.group(0)}' "
                "- descartar el intento anterior + la marca", s)
        elif RE_MARCA_DUDOSA.search(texto):
            add("INFO", "marca-corte", i, out_t,
                "contiene 'por eso' - WhisperX transcribe asi un 'borra eso' rapido, "
                "verificar de oido", s)
        for w in ws:
            if RE_FILLER.match(txt(w)):
                add("INFO", "filler", i, out_t + (float(w["start"]) - s),
                    f"filler vocal '{txt(w)}' como token suelto", float(w["start"]))

    # --- Check 3: gaps inter-range (Preset 5) ------------------------------
    #
    # OJO: un gap grande NO es un defecto por si mismo - es todo lo que se corto
    # a proposito (false starts, tomas malas, marcas "borra eso"). Marcar todo
    # gap >250ms daba 246 flags en un solo video, casi todos legitimos.
    # El defecto real es el gap que solo contiene SILENCIO: ahi el boundary se
    # puede apretar y no se hizo. Se distingue con el mapa de energia (o con los
    # words como fallback si el source no esta en disco).
    for i in range(len(ranges) - 1):
        a, b = ranges[i], ranges[i + 1]
        if a.get("source") != b.get("source"):
            continue
        ga, gb = float(a["end"]), float(b["start"])
        gap = gb - ga
        if gap <= GAP_INTER_MAX:
            continue
        db = energia.get(a.get("source"))
        if db is not None:
            voz = [(vs, ve) for vs, ve in runs_voz(db, ga, gb) if ve - vs >= 0.15]
            hay_voz = bool(voz)
        else:
            voz = None
            hay_voz = bool(words_de(transcripts, a.get("source"), ga, gb))

        if hay_voz:
            # Hay voz descartada. Solo interesa si es sustancial: puede ser una
            # frase buena tirada por error (Preset 3), no un false start.
            descartado = words_de(transcripts, a.get("source"), ga, gb)
            if len(descartado) >= 8:
                frag = " ".join(txt(w) for w in descartado[:10])
                add("INFO", "descarte", i, outs[i + 1],
                    f"{gap:.1f}s descartados con {len(descartado)} words: "
                    f"\"{frag}...\" - confirmar que era false start", ga)
            elif not descartado and voz:
                tot = sum(ve - vs for vs, ve in voz)
                if tot >= 0.5:
                    add("WARN", "descarte", i, outs[i + 1],
                        f"{tot:.1f}s de voz SIN words en el gap - "
                        "puede ser habla que el ASR omitio, no un descarte", ga)

    # --- Silencio audible en cada juntura del OUTPUT (el Preset 5 real) -----
    #
    # Lo que se oye entre dos ranges no es el gap del source (eso se elimino),
    # sino el tail del range i mas el head del range i+1. Ese es el residuo de
    # padding que se acumula: la medicion sobre 9 videos dio mediana 0.19s pero
    # con cola larga, y es la 2a fuente de tiempo muerto despues de las pausas.
    for i in range(len(ranges) - 1):
        if i not in heads_tails or (i + 1) not in heads_tails:
            continue
        tail = heads_tails[i][1]
        head = heads_tails[i + 1][0]
        junt = tail + head
        if junt > GAP_INTER_MAX:
            add("WARN", "juntura", i, outs[i + 1] - tail,
                f"{junt:.2f}s de silencio audible en la juntura "
                f"(tail {tail:.2f} + head {head:.2f}) - "
                f"apretar a last_word_end+0.05 / first_word_start-0.05",
                heads_tails[i][4])

    # --- Check 2: repeticiones lexicas (Preset 3) --------------------------
    for i in range(len(ranges) - 1):
        wa = words_de(transcripts, ranges[i].get("source"),
                      float(ranges[i]["start"]), float(ranges[i]["end"]))
        wb = words_de(transcripts, ranges[i + 1].get("source"),
                      float(ranges[i + 1]["start"]), float(ranges[i + 1]["end"]))
        if not wa or not wb:
            continue
        ta = [t for t in (norm_token(txt(w)) for w in wa) if t]
        tb = [t for t in (norm_token(txt(w)) for w in wb) if t]
        if not ta or not tb:
            continue
        n = 0
        for k in range(1, min(4, len(ta), len(tb)) + 1):
            if ta[:k] == tb[:k]:
                n = k
        if n:
            add("WARN", "repeticion", i, outs[i],
                f"ranges {i} y {i + 1} arrancan con {n} token(s) iguales "
                f"('{' '.join(ta[:n])}') - descartar el primer intento",
                float(ranges[i]["start"]))
        elif float(ranges[i + 1]["start"]) - float(ranges[i]["end"]) < 5.0:
            # sub-patron 3: frase abandonada + replay (comparten palabras clave, cerca)
            comunes = {t for t in ta if len(t) >= 6} & {t for t in tb if len(t) >= 6}
            if len(comunes) >= 2:
                add("INFO", "repeticion", i, outs[i],
                    f"ranges {i} y {i + 1} comparten {sorted(comunes)[:3]} "
                    "y estan a <5s - frase abandonada + replay?",
                    float(ranges[i]["start"]))

    flags.sort(key=lambda f: (SEV_ORDER[f["sev"]], f["out"]))
    return flags, cum


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Checklist automatico post-EDL")
    ap.add_argument("edl", type=Path)
    ap.add_argument("--json", type=Path, help="volcar el reporte a JSON")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 si hay ERROR (para encadenar antes de render)")
    ap.add_argument("--only", help="filtrar por check (coma-separado)")
    ap.add_argument("--summary", action="store_true",
                    help="solo el conteo por check, sin el detalle")
    args = ap.parse_args()

    if not args.edl.exists():
        print(f"no existe: {args.edl}")
        return 2

    edl, edit_dir, transcripts = cargar(args.edl)
    print(f"EDL: {args.edl}  ({len(edl.get('ranges', []))} ranges, "
          f"{len(transcripts)} transcript(s))")

    energia = {}
    for key, p in edl.get("sources", {}).items():
        fp = resolve_path(p, edit_dir)
        energia[key] = mapa_energia(fp, edit_dir / "verify") if fp.exists() else None

    flags, total = verificar(edl, edit_dir, transcripts, energia)

    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        flags = [f for f in flags if f["check"] in keep]

    n_err = sum(1 for f in flags if f["sev"] == "ERROR")
    n_wrn = sum(1 for f in flags if f["sev"] == "WARN")
    n_inf = len(flags) - n_err - n_wrn

    print(f"\nduracion de output: {fmt_t(total)}\n")
    if not flags:
        print("Sin flags. EDL listo para render.")
    elif args.summary:
        por_check = {}
        for f in flags:
            k = (f["sev"], f["check"])
            por_check[k] = por_check.get(k, 0) + 1
        for (sev, check), n in sorted(por_check.items(),
                                      key=lambda kv: (SEV_ORDER[kv[0][0]], -kv[1])):
            print(f"  {sev:<5} {check:<16} {n:>4}")
    else:
        actual = None
        for f in flags:
            if f["sev"] != actual:
                actual = f["sev"]
                print(f"\n===== {actual} =====")
            src = f" src {f['src']:8.2f}" if f["src"] is not None else " " * 13
            print(f"  out {fmt_t(f['out'])}{src}  [{f['check']}] r{f['range']}: {f['msg']}")

    print(f"\n{n_err} ERROR | {n_wrn} WARN | {n_inf} INFO")
    if n_err:
        print("Hay ERRORes: arreglarlos antes de renderizar preview (Preset 6).")

    if args.json:
        args.json.write_text(
            json.dumps(dict(edl=str(args.edl), total_s=total, flags=flags),
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"reporte -> {args.json}")

    return 1 if (args.strict and n_err) else 0


if __name__ == "__main__":
    sys.exit(main())
