#!/usr/bin/env python3
"""build_cleanup_edl.py - primer corte automatico (cleanup) de un video long-form.

Reemplaza a los generadores one-off que se quedaron en la PC vieja. Aplica las
reglas de limpieza del CLAUDE.md sobre el transcript de WhisperX, usando el MISMO
mapa de energia que verify_edl.py (-38 dB, ventanas de 50 ms):

  - marcas verbales ("borra eso", "corta eso", "cortalo"): descarta el intento
    anterior + la marca y conserva la retoma
  - "por eso" solo cuenta como marca si lo que sigue repite lo que venia antes
    (asi transcribe WhisperX un "borra eso" rapido, pero tambien es locucion)
  - fillers sueltos (mmm, ehh, amm), palabras truncadas con "..." que se retoman
    y tartamudeos de palabra repetida
  - repeticiones entre unidades (Preset 3): descarta el primer intento corto
  - corta TODA pausa >= 0.5 s (preferencia 2026-07-13). Las pausas se miden en el
    mapa de energia, no con el end declarado de las words (el ASR las estira)

Uso:
    uv run python helpers/build_cleanup_edl.py --edit-dir <edit> --source <video> [--key raw]

Salida: <edit>/edl.json + <edit>/cleanup_report.md (cada descarte con su motivo,
en output-time, para revisarlo en Studio). Despues: verify_edl.py y
studio_cuts.py to-studio.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import verify_edl as V  # noqa: E402  (mapa de energia, clamp de words, normalizacion)

HEAD, TAIL = 0.05, 0.08     # padding de cada corte (el residuo ES el ritmo deseado)
PAUSA_CORTE = 0.50          # toda pausa >= 0.5 s se corta
TOL_RUN = 0.30              # al extender el end por el run de voz, huecos tolerados
EXT_MAX = 1.00              # extension maxima del end mas alla de la ultima word
VENTANA_RETOMA = 45.0       # s hacia atras para buscar donde empezo el intento fallido
                            # (25 s no alcanzo: hablando mientras dibuja hubo 34 s entre
                            # el inicio del intento y la marca, 2026-10-07)
SILENCIO_LARGO = 8.0        # silencios cortados mas largos que esto van al reporte
STEP = V.VENTANA_MS / 1000.0
FIN_FRASE = re.compile(r"[.?!]$")


# ---------------------------------------------------------------------------
# Energia
# ---------------------------------------------------------------------------
def silencio_max(db, a: float, b: float) -> tuple[float, float, float]:
    """Silencio contiguo mas largo en [a,b] (blips <0.10 s cuentan como silencio)."""
    if b <= a:
        return 0.0, a, a
    voz = [r for r in V.runs_voz(db, a, b) if r[1] - r[0] >= V.BLIP_MAX]
    bordes = [a] + [x for r in voz for x in r] + [b]
    mejor = (0.0, a, a)
    for k in range(0, len(bordes) - 1, 2):
        q0, q1 = max(a, bordes[k]), min(b, bordes[k + 1])
        if q1 - q0 > mejor[0]:
            mejor = (q1 - q0, q0, q1)
    return mejor


def inicio_voz(db, t: float, piso: float) -> float:
    """Retrocede desde t mientras haya voz (max 0.25 s): el onset real de la word."""
    i = int(t / STEP)
    i_min = max(int(piso / STEP), int((t - 0.25) / STEP))
    while i - 1 >= i_min and db[i - 1] > V.VOZ_DB:
        i -= 1
    return min(t, i * STEP)


def fin_voz(db, t: float, techo: float) -> float:
    """Camina el run de voz desde t (tolerando huecos de TOL_RUN) hasta techo.

    Fix de la regresion 2026-08-04: los tokens normalizados ("4.5", ".env", siglas)
    traen el end truncado; el audio real sigue sonando despues del timestamp.
    """
    i = int(t / STEP)
    i_max = min(len(db), int(min(techo, t + EXT_MAX) / STEP))
    ultimo = None
    while i < i_max:
        if db[i] > V.VOZ_DB:
            ultimo = i
        elif ultimo is not None and (i - ultimo) * STEP > TOL_RUN:
            break
        elif ultimo is None and (i * STEP - t) > TOL_RUN:
            break
        i += 1
    return t if ultimo is None else max(t, (ultimo + 1) * STEP)


# ---------------------------------------------------------------------------
# Deteccion de descartes
# ---------------------------------------------------------------------------
def es_marca(toks: list[str], i: int) -> int:
    """Largo en tokens de la marca de corte que empieza en i (0 si no hay)."""
    t = toks[i]
    sig = toks[i + 1] if i + 1 < len(toks) else ""
    # norm_token quita acentos: "borré"/"corté" llegan como "borre"/"corte".
    # Variantes vistas con WhisperX (2026-10-07): "borrar eso", "borré eso", "borro eso"
    # y "borrazo" (las dos palabras pegadas en un solo token).
    if re.fullmatch(r"m*(borr[aoe]|borrar|cort[aoe]|cortar)", t) and sig == "eso":
        return 2
    if re.fullmatch(r"m*(borralo|cortalo|borrale|borrazo|borraeso)", t):
        return 1
    return 0


def buscar_inicio_intento(ws, toks, m0: int, retoma: list[str], kmin: int = 2) -> int | None:
    """Indice donde empieza el intento fallido: la ultima vez, antes de la marca,
    que aparecen los primeros k tokens de la retoma (k = 4..kmin)."""
    limite = float(ws[m0]["start"]) - VENTANA_RETOMA
    for k in range(min(4, len(retoma)), kmin - 1, -1):
        for p in range(m0 - 1, -1, -1):
            if float(ws[p]["start"]) < limite:
                break
            if toks[p:p + k] == retoma[:k]:
                return p
    # Intento de 1-3 palabras ("Ahora, borro eso. Ahora, actualmente..."): basta
    # con que la retoma arranque igual que una de las ultimas 3 words
    if retoma:
        for p in range(m0 - 1, max(-1, m0 - 4), -1):
            if toks[p] == retoma[0]:
                return p
    return None


def inicio_frase(ws, cov, m0: int) -> int:
    """Fallback: retrocede hasta el inicio de la frase (puntuacion o pausa >= 1 s)."""
    limite = float(ws[m0]["start"]) - 20.0
    q = m0 - 1
    while q - 1 >= 0:
        if FIN_FRASE.search(V.txt(ws[q - 1])):
            break
        if cov[q][0] - cov[q - 1][1] >= 1.0 or float(ws[q - 1]["start"]) < limite:
            break
        q -= 1
    return max(q, 0)


def detectar(ws, cov, toks, db):
    """Devuelve {indice: motivo} de words a descartar + lista de avisos."""
    n = len(ws)
    desc: dict[int, str] = {}
    avisos: list[str] = []

    def retoma_desde(j: int) -> list[str]:
        out = []
        while j < n and len(out) < 4:
            if not V.RE_FILLER.match(V.txt(ws[j])) and toks[j]:
                out.append(toks[j])
            j += 1
        return out

    # 1) Marcas verbales
    i = 0
    while i < n:
        L = es_marca(toks, i)
        dudosa = False
        if not L and toks[i] == "por" and i + 1 < n and toks[i + 1] == "eso":
            L, dudosa = 2, True
        if not L:
            i += 1
            continue
        m0, m1 = i, i + L - 1
        while m1 + 1 < n and toks[m1 + 1] == "eso":  # "Borra eso. eso." (eco del ASR)
            m1 += 1
        retoma = retoma_desde(m1 + 1)
        p = buscar_inicio_intento(ws, toks, m0, retoma)
        if dudosa:
            # "por eso" solo es marca si la retoma repite lo de antes y esta cerca
            if p is None or float(ws[m0]["start"]) - float(ws[p]["start"]) > 15.0:
                i += L
                continue
            motivo = "marca 'por eso' (= 'borra eso' rapido: lo que sigue repite lo anterior)"
        else:
            motivo = f"marca '{' '.join(V.txt(w) for w in ws[m0:m1 + 1])}'"
        if p is None:
            p = inicio_frase(ws, cov, m0)
            avisos.append(f"{V.fmt_t(float(ws[m0]['start']))} marca sin retoma clara: "
                          f"se descarta desde el inicio de la frase - revisar")
        for j in range(p, m1 + 1):
            desc.setdefault(j, f"intento descartado por {motivo}" if j < m0 else motivo)
        i = m1 + 1

    # 2) Fillers, truncadas y tartamudeos
    for j in range(n):
        if j in desc:
            continue
        t = V.txt(ws[j])
        if V.RE_FILLER.match(t):
            desc[j] = f"filler '{t}'"
            continue
        trunc = t.endswith("...") or t.endswith("…")
        if (trunc and j > 0 and toks[j] and toks[j - 1] == toks[j]
                and V.txt(ws[j - 1]).endswith(("...", "…"))):
            desc[j] = f"titubeo '{V.txt(ws[j - 1])} {t}'"  # "con... con..."
            continue
        if trunc and len(toks[j]) >= 3:
            sig = [toks[k] for k in range(j + 1, min(n, j + 4)) if k not in desc]
            if any(s.startswith(toks[j][:3]) for s in sig):
                desc[j] = f"palabra truncada '{t}' que se retoma despues"
                continue
        if (j + 1 < n and toks[j] and toks[j] == toks[j + 1]
                and cov[j + 1][0] - cov[j][1] < 1.0
                and not FIN_FRASE.search(t)):
            desc[j] = f"tartamudeo '{t} {V.txt(ws[j + 1])}'"
    return desc, avisos


def unidades(idx_vivos, cov, db, desc_set):
    """Agrupa words vivas en unidades: se corta en cada descarte y en pausa >= 0.5 s."""
    grupos, actual = [], []
    for j in idx_vivos:
        if actual:
            prev = actual[-1]
            hay_descarte = any(k in desc_set for k in range(prev + 1, j))
            pausa = silencio_max(db, cov[prev][1], cov[j][0])[0]
            # Un hueco declarado >= 0.5 s se percibe como pausa aunque el silencio
            # a -38 dB mida 0.40-0.49 (las colas de las words caen bajo el umbral):
            # verify_edl lo marca como ERROR pausa-intra. Se corta si no hay voz dentro.
            hueco = cov[j][0] - cov[prev][1]
            voz = sum(e - s for s, e in V.runs_voz(db, cov[prev][1], cov[j][0])
                      if e - s >= V.BLIP_MAX)
            pausa_percibida = hueco >= PAUSA_CORTE and pausa >= 0.30 and voz < 0.15
            if hay_descarte or pausa >= PAUSA_CORTE or pausa_percibida:
                grupos.append(actual)
                actual = []
        actual.append(j)
    if actual:
        grupos.append(actual)
    return grupos


def repeticiones(grupos, ws, toks, cov, desc):
    """Preset 3: unidad i abandonada y retomada por la unidad i+1."""
    avisos = []
    for a, b in zip(grupos, grupos[1:]):
        if cov[b[0]][0] - cov[a[-1]][1] > 5.0:
            continue
        ta = [toks[j] for j in a if toks[j]]
        tb = [toks[j] for j in b if toks[j]]
        if not ta or not tb:
            continue
        k = 0
        while k < min(4, len(ta), len(tb)) and ta[k] == tb[k]:
            k += 1
        frase = " ".join(V.txt(ws[j]) for j in a)
        abierta = not FIN_FRASE.search(V.txt(ws[a[-1]]))
        prefijo = tb[:len(ta)] == ta
        # Reinicio = el primer intento queda contenido en el segundo, o se desvia
        # en a lo mas 1 token. Si ambos siguen distinto es paralelismo legitimo
        # ("este corre en tu PC, este corre en la nube"): solo aviso.
        reinicio = prefijo or (abierta and len(ta) - k <= 1)
        if (k >= 2 and len(ta) <= 12 and reinicio) or (k >= 1 and len(ta) <= 2):
            for j in a:
                desc.setdefault(j, f"repeticion: '{frase}' se retoma enseguida")
        elif k >= 2:
            avisos.append(f"{V.fmt_t(cov[a[0]][0])} posible repeticion larga (no se corto): "
                          f"'{frase[:80]}'")
    return avisos


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Primer corte automatico (cleanup)")
    ap.add_argument("--edit-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="video fuente")
    ap.add_argument("--key", default="raw", help="source key del EDL = transcripts/<key>.json")
    args = ap.parse_args()

    edit = args.edit_dir.resolve()
    tp = edit / "transcripts" / f"{args.key}.json"
    d = json.loads(tp.read_text(encoding="utf-8"))
    ws = sorted((w for w in d["words"] if V.es_palabra(w)), key=lambda w: float(w["start"]))
    db = V.mapa_energia(args.source.resolve(), edit / "verify")
    if db is None:
        sys.exit("sin mapa de energia: no se puede cortar por pausas")

    toks = [V.norm_token(V.txt(w)) for w in ws]
    cov = [V.cobertura(w, db) for w in ws]

    desc, avisos = detectar(ws, cov, toks, db)
    vivos = [j for j in range(len(ws)) if j not in desc]
    grupos = unidades(vivos, cov, db, set(desc))
    avisos += repeticiones(grupos, ws, toks, cov, desc)

    # Decisiones editoriales que el regex no puede tomar ("borra toda esa ultima
    # seccion", "borra eso porque no estaba grabando la pantalla"...): las escribe
    # quien lee el transcript, por tiempo de SOURCE, en <edit>/cleanup_overrides.json
    #   {"drop": [{"desde": s, "hasta": s, "motivo": "..."}], "keep": [...]}
    # Se aplican por el centro de la cobertura de cada word.
    ov_path = edit / "cleanup_overrides.json"
    if ov_path.exists():
        ov = json.loads(ov_path.read_text(encoding="utf-8"))
        for j in range(len(ws)):
            m = (cov[j][0] + cov[j][1]) / 2
            for o in ov.get("keep", []):
                if o["desde"] <= m <= o["hasta"]:
                    desc.pop(j, None)
            for o in ov.get("drop", []):
                if o["desde"] <= m <= o["hasta"]:
                    desc[j] = f"decision editorial: {o['motivo']}"
    vivos = [j for j in range(len(ws)) if j not in desc]
    grupos = unidades(vivos, cov, db, set(desc))

    # --- ranges con boundaries por energia -------------------------------
    ranges = []
    for g in grupos:
        a, b = g[0], g[-1]
        piso = cov[a - 1][1] + 0.02 if a > 0 else 0.0
        techo = cov[b + 1][0] - 0.02 if b + 1 < len(ws) else len(db) * STEP
        s = max(piso, inicio_voz(db, cov[a][0], piso) - HEAD)
        e = min(techo, fin_voz(db, cov[b][1], techo) + TAIL)
        if ranges and s <= ranges[-1]["end"] + 0.02:
            ranges[-1]["end"] = round(e, 3)
            ranges[-1]["_words"] += g
            continue
        ranges.append({"source": args.key, "start": round(s, 3), "end": round(e, 3), "_words": list(g)})

    for r in ranges:
        texto = " ".join(V.txt(ws[j]) for j in r.pop("_words"))
        r["beat"] = texto if len(texto) <= 90 else texto[:87] + "..."

    edl = {"sources": {args.key: str(args.source.resolve()).replace("\\", "/")},
           "grade": None, "ranges": ranges}
    out = edit / "edl.json"
    if out.exists():
        bk = edit / "edl_backups"
        bk.mkdir(exist_ok=True)
        n = len(list(bk.glob("edl_cleanup_*.json")))
        (bk / f"edl_cleanup_{n:02d}.json").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    out.write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- reporte ---------------------------------------------------------
    outs, cum = [], 0.0
    for r in ranges:
        outs.append(cum)
        cum += r["end"] - r["start"]

    def out_de(t: float) -> float:
        """Output-time del punto de corte mas cercano a t (source)."""
        for k, r in enumerate(ranges):
            if r["start"] >= t:
                return outs[k]
        return cum

    lineas = [f"# Reporte de cleanup — {args.source.name}", "",
              f"- Fuente: {V.fmt_t(len(db) * STEP)} → salida **{V.fmt_t(cum)}** "
              f"({len(ranges)} cortes, {len(desc)} words descartadas de {len(ws)})", ""]
    # descartes agrupados por tramos consecutivos con el mismo motivo
    lineas += ["## Descartes (output-time = donde queda el corte)", ""]
    bloque = None
    for j in sorted(desc):
        if bloque and j == bloque[1] + 1 and desc[j].split(":")[0] == desc[bloque[0]].split(":")[0]:
            bloque[1] = j
            continue
        if bloque:
            lineas.append(_linea(bloque, ws, desc, out_de))
        bloque = [j, j]
    if bloque:
        lineas.append(_linea(bloque, ws, desc, out_de))

    largos = []
    for k in range(len(ranges) - 1):
        g = ranges[k + 1]["start"] - ranges[k]["end"]
        if g >= SILENCIO_LARGO:
            largos.append(f"- out {V.fmt_t(outs[k + 1])} · src {V.fmt_t(ranges[k]['end'])} · "
                          f"{g:.0f} s cortados — revisar si en pantalla pasa algo que se deba ver")
    if largos:
        lineas += ["", "## Tramos largos cortados (posible demo en pantalla sin voz)", ""] + largos
    if avisos:
        lineas += ["", "## Avisos", ""] + [f"- {a}" for a in avisos]
    (edit / "cleanup_report.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    print(f"fuente {V.fmt_t(len(db) * STEP)} -> salida {V.fmt_t(cum)} | {len(ranges)} ranges | "
          f"{len(desc)} words descartadas | {len(largos)} silencios largos | {len(avisos)} avisos")
    print(f"-> {out}\n-> {edit / 'cleanup_report.md'}")
    return 0


def _linea(bloque, ws, desc, out_de) -> str:
    a, b = bloque
    t0 = float(ws[a]["start"])
    texto = " ".join(V.txt(w) for w in ws[a:b + 1])
    return (f"- out {V.fmt_t(out_de(t0))} · src {V.fmt_t(t0)} · {desc[a].split(':')[0]}: "
            f"\"{texto[:100]}\"")


if __name__ == "__main__":
    sys.exit(main())
