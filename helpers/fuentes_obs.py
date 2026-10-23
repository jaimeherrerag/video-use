#!/usr/bin/env python3
"""fuentes_obs.py - grabacion separada de OBS (desde 2026-10-23): guia + camara + pantalla.

OBS graba tres archivos a la vez en D:/Footage/grabaciones:
  <fecha>.mp4           la GUIA: el programa a 1280x720 con el audio PCM (pista 1 mic, 2 escritorio).
                        Es el reloj del video: transcript, cortes, auditores y karaoke van sobre ella.
  <fecha> camara.mp4    la camara 4K nativa (plugin Source Record), con las pistas del mezclador en AAC
  <fecha> pantalla.mp4  la pantalla 1080p nativa (Source Record), igual; se graba TODO el tiempo

Los filtros de Source Record arrancan unos cuadros despues de la guia (5-7 medidos): el desfase se mide
por audio (la pista 1, el mic, esta en los tres). El log de OBS anota cada cambio de escena con su hora;
de ahi sale la VISTA de cada corte, o sea de que archivo sale la imagen:

  camara           la camara a cuadro completo            (escena "webcam full")
  pantalla         la pantalla a cuadro completo          (escena "Solo pantalla")
  pantalla_camara  la pantalla con la camara en un circulo (escena "Pantalla + webcam"), como en OBS

    ingesta --guia G --edit-dir E   archivos hermanos, mic16k.wav, desfases, escenas del log y el circulo
                                    de la coleccion de escenas -> E/fuentes.json (+ copia del log)
    vistas  --edit-dir E            despues de build_cleanup_edl: agrega las fuentes al edl.json, parte los
                                    cortes donde cambio la escena y le pone su vista a cada uno

Los cortes siguen en el tiempo de la guia (source = la guia). render.py y studio_maestro.py toman la imagen
del archivo de su vista en t - desfase, y el audio siempre de la guia.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

FPS = 30
SALIDA = (1920, 1080)                           # lienzo de un largo
SUFIJOS = {"camara": " camara", "pantalla": " pantalla"}
# circulo de la escena "Pantalla + webcam" medido el 2026-10-23 (1080p): respaldo si no se puede leer OBS
PIP_DEFECTO = {"cx": 1684, "cy": 893, "d": 318, "recorte": {"x": 1011, "y": 0, "w": 2160, "h": 2160}}
OBS = Path(os.environ.get("APPDATA", "")) / "obs-studio"


def vista_de_escena(nombre: str) -> str:
    """Vista por defecto segun el nombre de la escena (se puede corregir a mano en fuentes.json)."""
    n = nombre.lower()
    pantalla = re.search(r"pantal+a|screen|escritorio", n)    # la escena de Jaime se llama "Pantallla + webcam"
    if pantalla and ("webcam" in n or "cam" in n):
        return "pantalla_camara"
    if pantalla:
        return "pantalla"
    return "camara"


def archivo_de_vista(vista: str) -> str:
    """De que archivo sale la imagen BASE de una vista."""
    return "camara" if vista == "camara" else "pantalla"


# ── audio y desfases ──────────────────────────────────────────────────────────

def mono16k(src: Path, segundos: float | None = None) -> np.ndarray:
    """Pista 1 (el mic) de un archivo como mono 16 kHz en float32."""
    t = ["-t", f"{segundos:.1f}"] if segundos else []
    out = subprocess.run(["ffmpeg", "-v", "error", *t, "-i", str(src), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768


def desfase(guia: np.ndarray, otro: np.ndarray, sr: int = 16000, maximo: float = 3.0) -> tuple[float, float]:
    """Segundos que van de la guia al otro archivo (t_otro = t_guia - desfase) y la correlacion del pico."""
    n = 1 << int(np.ceil(np.log2(len(guia) + len(otro))))
    cc = np.fft.irfft(np.fft.rfft(guia, n) * np.conj(np.fft.rfft(otro, n)), n)
    m = int(maximo * sr)
    cand = np.concatenate([cc[:m], cc[-m:]])
    lags = np.concatenate([np.arange(m), np.arange(-m, 0)])
    i = int(np.argmax(cand))
    return lags[i] / sr, float(cand[i] / (np.sqrt((guia ** 2).sum() * (otro ** 2).sum()) + 1e-9))


# ── log de OBS: escenas ───────────────────────────────────────────────────────

def _seg(hhmmss: str) -> float:
    h, m, s = hhmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def buscar_log(guia: Path) -> Path | None:
    """El log de OBS que escribio este archivo (el nombre aparece en la linea 'Writing ... file')."""
    nombre = guia.name
    logs = sorted((OBS / "logs").glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for lg in logs:
        try:
            if nombre in lg.read_text(encoding="utf-8", errors="replace"):
                return lg
        except OSError:
            continue
    return None


def escenas_del_log(log: Path, guia: Path) -> tuple[list[dict], str | None]:
    """Escenas al aire durante la grabacion de `guia`, en segundos de la guia: [{desde, escena}].
    Devuelve tambien el nombre de la coleccion de escenas que estaba cargada."""
    lineas = log.read_text(encoding="utf-8", errors="replace").splitlines()
    rx_t = re.compile(r"^(\d\d:\d\d:\d\d\.\d+): (.*)$")
    rx_esc = re.compile(r"(?:User s|S)witched to scene '(.*)'")
    escena, coleccion, t0, eventos, grabando = None, None, None, [], False
    previo = None
    for ln in lineas:
        m = rx_t.match(ln)
        if not m:
            continue
        t, txt = _seg(m.group(1)), m.group(2)
        if previo is not None and t < previo - 3600:      # paso la medianoche
            t += 86400
        previo = t
        mc = re.search(r"Loading stream configuration|Scene collection: '?(.*?)'?$", txt)
        if "Switched to scene collection" in txt or txt.startswith("Loading scene collection"):
            coleccion = txt.split("'")[1] if "'" in txt else coleccion
        me = rx_esc.search(txt)
        if me:
            if grabando:
                eventos.append({"desde": round(t - t0, 3), "escena": me.group(1)})
            else:
                escena = me.group(1)
        if "==== Recording Start" in txt:
            t_inicio = t
        if "Writing" in txt and guia.name in txt:
            grabando, t0 = True, t_inicio
            eventos = [{"desde": 0.0, "escena": escena or "?"}]
        elif grabando and "==== Recording Stop" in txt:
            break
    return eventos, coleccion


# ── coleccion de escenas: el circulo de la camara ───────────────────────────

def pip_de_obs(escena_pip: str | None) -> dict:
    """Centro y diametro del circulo (en el lienzo 1080p) y el recorte de la camara que se ve adentro,
    leidos de la coleccion de escenas de OBS: la escena del pip tiene una escena anidada con filtro de
    mascara (PNG con alfa) que a su vez contiene la camara. Si algo no cuadra, el circulo medido."""
    try:
        from PIL import Image
        cols = sorted((OBS / "basic" / "scenes").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        d = json.loads(cols[0].read_text(encoding="utf-8"))
        lw, lh = d["resolution"]["x"], d["resolution"]["y"]
        fuentes = {s["name"]: s for s in d["sources"]}
        esc = fuentes[escena_pip]
        item = next(it for it in esc["settings"]["items"]
                    if it.get("visible", True) and fuentes[it["name"]]["id"] == "scene"
                    and any(f["id"] == "mask_filter" for f in fuentes[it["name"]].get("filters", [])))
        anidada = fuentes[item["name"]]
        mascara = next(f for f in anidada["filters"] if f["id"] == "mask_filter")["settings"]["image_path"]
        a = np.asarray(Image.open(mascara).convert("RGBA"))[..., 3]
        ys, xs = np.nonzero(a > 127)
        mx0, mx1, my0, my1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        km = lw / a.shape[1]                     # la mascara cubre la escena anidada (del tamano del lienzo)
        # camara dentro de la escena anidada (puede venir en otra escena anidada mas)
        off_x, off_y, esc_cam = 0.0, 0.0, anidada
        for _ in range(3):
            it = next(i for i in esc_cam["settings"]["items"] if i.get("visible", True))
            off_x, off_y = off_x + it["pos"]["x"], off_y + it["pos"]["y"]
            sig = fuentes[it["name"]]
            if sig["id"] != "scene":
                break
            esc_cam = sig
        sx, sy = item["scale"]["x"], item["scale"]["y"]
        k = SALIDA[0] / lw
        cx = (item["pos"]["x"] + (mx0 + mx1) / 2 * km * sx) * k
        cy = (item["pos"]["y"] + (my0 + my1) / 2 * km * sy) * k
        dm = (mx1 - mx0) * km * sx * k
        rec = {"x": int(round(mx0 * km - off_x)), "y": int(round(my0 * km - off_y)),
               "w": int(round((mx1 - mx0) * km)), "h": int(round((my1 - my0) * km))}
        return {"cx": int(round(cx)), "cy": int(round(cy)), "d": int(round(dm)) // 2 * 2, "recorte": rec,
                "de": f"OBS: escena '{escena_pip}', mascara {Path(mascara).name}"}
    except Exception as e:                              # noqa: BLE001  (OBS cambia de formato: el medido)
        print(f"  aviso: no pude leer el circulo de OBS ({e}); uso el medido el 2026-10-23")
        return dict(PIP_DEFECTO, de="medido 2026-10-23")


# ── comandos ──────────────────────────────────────────────────────────────────

def hermano(guia: Path, sufijo: str) -> Path | None:
    for ext in (".mp4", ".mkv", ".mov"):
        p = guia.with_name(guia.stem + sufijo + ext)
        if p.exists():
            return p
    return None


def ingesta(guia: Path, edit: Path) -> None:
    edit.mkdir(parents=True, exist_ok=True)
    archivos = {"guia": guia}
    for k, suf in SUFIJOS.items():
        p = hermano(guia, suf)
        if not p:
            sys.exit(f"no encontre '{guia.stem}{suf}.mp4' junto a la guia")
        archivos[k] = p
    for k, p in archivos.items():
        print(f"  {k:9s} {p.name}  ({p.stat().st_size / 1e9:.2f} GB)")

    # mic de la guia a 16 kHz: lo que usan transcribe_local, verify_edl y audit_bordes
    wav = edit / "mic16k.wav"
    if not wav.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(guia), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        print(f"  mic16k.wav desde la guia")

    # desfases por audio (los primeros 2 min bastan: el desfase es constante)
    g = mono16k(guia, 120)
    desf = {}
    for k in SUFIJOS:
        d, c = desfase(g, mono16k(archivos[k], 120))
        if c < 0.5:
            sys.exit(f"el audio de {archivos[k].name} no se parece al de la guia (correlacion {c:.2f}): "
                     "revisa que el filtro grabe la pista 1 (o All)")
        desf[k] = round(d, 4)
        print(f"  desfase {k}: {d * 1000:+.1f} ms ({d * FPS:+.1f} cuadros, correlacion {c:.2f})")

    # escenas del log de OBS
    log = buscar_log(guia)
    escenas, mapa_esc = [], {}
    if log:
        shutil.copy2(log, edit / "obs_log.txt")
        escenas, _ = escenas_del_log(log, guia)
        for e in escenas:
            mapa_esc.setdefault(e["escena"], vista_de_escena(e["escena"]))
            e["vista"] = mapa_esc[e["escena"]]
        print(f"  log de OBS: {log.name} -> {len(escenas)} tramos de escena")
        for e in escenas:
            print(f"    {e['desde']:8.2f}s  {e['escena']}  -> {e['vista']}")
    else:
        print("  aviso: no encontre el log de OBS de esta grabacion: todo queda en vista 'camara' "
              "(se puede editar fuentes.json o cambiar la vista en Studio)")
        escenas = [{"desde": 0.0, "escena": "?", "vista": "camara"}]

    escena_pip = next((e for e, v in mapa_esc.items() if v == "pantalla_camara"), None)
    pip = pip_de_obs(escena_pip) if escena_pip else dict(PIP_DEFECTO, de="medido 2026-10-23")
    print(f"  circulo: centro ({pip['cx']}, {pip['cy']}), {pip['d']} px; camara {pip['recorte']} ({pip['de']})")

    f = {"guia": str(guia).replace("\\", "/"),
         "archivos": {k: str(p).replace("\\", "/") for k, p in archivos.items() if k != "guia"},
         "desfase": desf, "escenas": escenas, "vistas_por_escena": mapa_esc, "pip": pip,
         "log": "obs_log.txt" if log else None, "generado": datetime.now().isoformat(timespec="seconds")}
    (edit / "fuentes.json").write_text(json.dumps(f, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {edit / 'fuentes.json'}")


def vista_en(escenas: list[dict], t: float) -> str:
    v = escenas[0]["vista"]
    for e in escenas:
        if e["desde"] <= t + 1e-6:
            v = e["vista"]
    return v


def vistas(edit: Path) -> None:
    f = json.loads((edit / "fuentes.json").read_text(encoding="utf-8"))
    edl_p = edit / "edl.json"
    edl = json.loads(edl_p.read_text(encoding="utf-8"))
    claves = [k for k, v in edl["sources"].items() if Path(v).name == Path(f["guia"]).name]
    if not claves:
        sys.exit("el edl.json no usa la guia de fuentes.json como fuente")
    kg = claves[0]
    # las vistas pueden haberse corregido a mano en vistas_por_escena
    escenas = [dict(e, vista=f["vistas_por_escena"].get(e["escena"], e.get("vista", "camara"))) for e in f["escenas"]]
    cambios = [e["desde"] for e in escenas[1:]]

    nuevos, n_part = [], 0
    for r in edl["ranges"]:
        if r["source"] != kg:
            nuevos.append(r)
            continue
        # se parte en cada cambio de escena (en la rejilla de cuadros) si los dos lados duran >= 0.4 s; un
        # pedacito menor no vale un corte: el corte entero se queda con la vista que mas dura
        cortes = []
        for c in cambios:
            c = round(round(c * FPS) / FPS, 4)
            if r["start"] + 0.4 <= c <= r["end"] - 0.4 and (not cortes or c - cortes[-1] >= 0.4):
                cortes.append(c)
        if not cortes and any(r["start"] < c < r["end"] for c in cambios):
            dur = {}
            ts = [r["start"], *[c for c in cambios if r["start"] < c < r["end"]], r["end"]]
            for a_, b_ in zip(ts, ts[1:]):
                v = vista_en(escenas, a_ + 1e-3)
                dur[v] = dur.get(v, 0) + b_ - a_
            nuevos.append(dict(r, vista=max(dur, key=dur.get)))
            continue
        bordes = [r["start"], *cortes, r["end"]]
        for j in range(len(bordes) - 1):
            p = dict(r, start=round(bordes[j], 4), end=round(bordes[j + 1], 4))
            p["vista"] = vista_en(escenas, (p["start"] + p["end"]) / 2 if j == 0 and not cortes else p["start"] + 0.01)
            if j > 0:
                p["continuo"] = True              # mismo audio sin hueco: sin fundidos en esta juntura
                n_part += 1
            elif "continuo" in r:
                p["continuo"] = r["continuo"]
            nuevos.append(p)

    bk = edit / "edl_backups"
    bk.mkdir(exist_ok=True)
    shutil.copy2(edl_p, bk / f"edl_antes-vistas_{datetime.now():%Y%m%d-%H%M%S}.json")
    edl["sources"].update(f["archivos"])
    edl["fuentes"] = {"guia": kg, "desfase": f["desfase"], "pip": f["pip"]}
    edl["ranges"] = nuevos
    edl_p.write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
    cuenta = {}
    for r in nuevos:
        cuenta[r.get("vista", "-")] = cuenta.get(r.get("vista", "-"), 0) + 1
    print(f"  {len(nuevos)} cortes ({n_part} partidos en un cambio de escena): "
          + ", ".join(f"{v} {n}" for v, n in sorted(cuenta.items())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ingesta", "vistas"])
    ap.add_argument("--guia", type=Path, help="ingesta: el archivo principal de OBS (la guia de 720p)")
    ap.add_argument("--edit-dir", required=True, type=Path)
    a = ap.parse_args()
    edit = a.edit_dir.resolve()
    if a.cmd == "ingesta":
        if not a.guia:
            ap.error("ingesta necesita --guia")
        ingesta(a.guia.resolve(), edit)
    else:
        vistas(edit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
