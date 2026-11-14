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
    piezas  --origen L --piezas P --edit-dir E
                                    SHORT DESDE UN LARGO (estandar): pega los pedazos de P (segundos de la guia
                                    del largo L) en guia/camara/pantalla con el mismo reloj, trae los cortes
                                    aprobados del largo y los encuadra en 50/50 o camara completa segun la escena

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

def mono16k(src: Path, segundos: float | None = None, desde: float = 0.0) -> np.ndarray:
    """Pista 1 (el mic) de un archivo como mono 16 kHz en float32."""
    t = (["-t", f"{segundos:.1f}"] if segundos else []) + (["-ss", f"{desde:.2f}"] if desde else [])
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


def duracion(src: Path, stream: str = "a:0") -> float:
    """Duracion exacta de un stream (por defecto el mic): la del contenedor puede ser la del video."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
                          "stream=duration:format=duration", "-of", "json", str(src)],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = (d.get("streams") or [{}])[0].get("duration")
    return float(s if s not in (None, "N/A") else d["format"]["duration"])


def fuente_en(fuentes: dict, k: str, t: float) -> tuple[str, float] | None:
    """(clave de la fuente, desfase) de la `camara` o la `pantalla` en el segundo `t` de la guia. Con varios
    tramos (pausas que corren el Source Record, o varias sesiones juntas) cada uno trae su archivo y su desfase;
    None = en ese tramo no hay archivo de ese tipo (la imagen sale de la guia). Sin tramos, el valor unico."""
    tramos = fuentes.get("tramos")
    if not tramos:
        return k, float(fuentes["desfase"].get(k, 0.0))
    v = tramos[0][k]
    for tr in tramos:
        if tr["desde"] <= t + 1e-6:
            v = tr[k]
    return (v[0], float(v[1])) if v else None


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


def escenas_del_log(log: Path, guia: Path) -> tuple[list[dict], str | None, list[float], dict[str, float]]:
    """Escenas al aire durante la grabacion de `guia`, en segundos de la guia: [{desde, escena}].
    Devuelve tambien el nombre de la coleccion de escenas que estaba cargada, el segundo de cada pausa y el
    segundo de la guia en que Source Record abrio cada uno de sus archivos ({nombre: s})."""
    lineas = log.read_text(encoding="utf-8", errors="replace").splitlines()
    rx_t = re.compile(r"^(\d\d:\d\d:\d\d\.\d+): (.*)$")
    rx_esc = re.compile(r"(?:User s|S)witched to scene '(.*)'")
    escena, coleccion, t0, eventos, grabando, inicios = None, None, None, [], False, {}
    previo = None
    # la pausa de OBS no avanza el archivo: se resta, y un cambio en pausa entra donde se pauso
    pausado_desde, en_pausa, pausas = None, 0.0, []
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
        if grabando and re.search(r"output adv_file_output paused", txt):
            pausado_desde = t
            pausas.append(round(t - t0 - en_pausa, 3))
        elif grabando and re.search(r"output adv_file_output unpaused", txt) and pausado_desde is not None:
            en_pausa += t - pausado_desde
            pausado_desde = None
        me = rx_esc.search(txt)
        if me:
            if grabando:
                desde = round((pausado_desde if pausado_desde is not None else t) - t0 - en_pausa, 3)
                if eventos and eventos[-1]["desde"] == desde:
                    eventos.pop()                       # varios cambios en una misma pausa: cuenta el ultimo
                eventos.append({"desde": desde, "escena": me.group(1)})
            else:
                escena = me.group(1)
        if "==== Recording Start" in txt:
            t_inicio = t
        if "Writing" in txt and guia.name in txt:
            grabando, t0 = True, t_inicio
            eventos = [{"desde": 0.0, "escena": escena or "?"}]
        elif grabando and "Writing" in txt and "'" in txt:
            # Source Record abre su archivo cuando la fuente sale al aire: si fue despues, aqui queda en que
            # segundo de la guia empezo (Higgsfield 2026-11-05: la pantalla, con la primera escena de pantalla)
            inicios[Path(txt.split("'")[-2]).name] = round((pausado_desde if pausado_desde is not None else t)
                                                           - t0 - en_pausa, 3)
        elif grabando and "==== Recording Stop" in txt:
            break
    return eventos, coleccion, pausas, inicios


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
        # la coleccion es la de AHORA: si el lienzo cambio despues de grabar (vertical para shorts), no sirve
        if abs(lw / lh - SALIDA[0] / SALIDA[1]) > 0.01:
            raise ValueError(f"el lienzo actual de OBS es {lw}x{lh}, no 16:9")
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

def _hora(stem: str) -> datetime | None:
    m = re.match(r"(\d{4}-\d\d-\d\d \d\d-\d\d-\d\d)", stem)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H-%M-%S") if m else None


def hermano(guia: Path, sufijo: str) -> tuple[Path, float] | None:
    """El archivo de Source Record de esta guia y los segundos (de reloj) que arranco despues. Source Record
    puede nombrarlo con un segundo de mas, o arrancar minutos despues si el filtro se activo tarde (Higgsfield,
    2026-11-05: la pantalla empezo 3.5 min despues): se toma el mas cercano que empiece en los 10 min siguientes."""
    for ext in (".mp4", ".mkv", ".mov"):
        p = guia.with_name(guia.stem + sufijo + ext)
        if p.exists():
            return p, 0.0
    h0 = _hora(guia.stem)
    if not h0:
        return None
    cand = []
    for p in guia.parent.glob(f"*{sufijo}.*"):
        h = _hora(p.stem)
        if h and p.suffix in (".mp4", ".mkv", ".mov") and 0 <= (h - h0).total_seconds() <= 600:
            cand.append(((h - h0).total_seconds(), p))
    if not cand:
        return None
    s, p = min(cand)
    return p, s


def ingesta(guia: Path, edit: Path) -> None:
    edit.mkdir(parents=True, exist_ok=True)
    archivos, tarde = {"guia": guia}, {}
    for k, suf in SUFIJOS.items():
        h = hermano(guia, suf)
        if not h:
            sys.exit(f"no encontre '{guia.stem}{suf}.mp4' junto a la guia")
        archivos[k], tarde[k] = h
        if tarde[k] > 3:
            print(f"  aviso: {k} arranco ~{tarde[k]:.0f} s despues de la guia: antes de eso no hay imagen de {k}")
    for k, p in archivos.items():
        print(f"  {k:9s} {p.name}  ({p.stat().st_size / 1e9:.2f} GB)")

    # mic de la guia a 16 kHz: lo que usan transcribe_local, verify_edl y audit_bordes
    wav = edit / "mic16k.wav"
    if not wav.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(guia), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        print(f"  mic16k.wav desde la guia")

    # escenas del log de OBS
    log = buscar_log(guia)
    escenas, mapa_esc, pausas, del_log = [], {}, [], set()
    if log:
        shutil.copy2(log, edit / "obs_log.txt")
        escenas, _, pausas, inicios = escenas_del_log(log, guia)
        # un archivo que arranco tarde: el log dice en que segundo de la guia (la hora del nombre es de reloj y
        # no descuenta las pausas)
        for k in SUFIJOS:
            if tarde[k] > 3 and archivos[k].name in inicios:
                tarde[k] = inicios[archivos[k].name]
                del_log.add(k)
                print(f"  {k} empieza en el segundo {tarde[k]:.2f} de la guia (log de OBS)")
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

    # desfases por audio, uno por tramo entre pausas. Cada pausa puede correr el Source Record respecto a la guia:
    # 1 cuadro en el largo del segundo cerebro (2026-10-26), pero hasta 2.4 s en el de Higgsfield (2026-11-06), asi
    # que un solo valor no sirve: `tramos` guarda el desfase de cada tramo y render/Studio lo leen con fuente_en().
    # Cada tramo se mide sobre el mismo (no a ciegas desde la pausa) y la busqueda va alrededor del anterior.
    dur_g = duracion(guia)
    # bordes: las pausas y el inicio de un archivo que arranco tarde (antes de el, ese tipo queda sin imagen)
    bordes = sorted({0.0, *[p for p in pausas if 1.0 < p < dur_g - 1.0],
                     *[tarde[k] for k in del_log if 1.0 < tarde[k] < dur_g - 1.0]}) + [dur_g]
    desf, por_tipo = {}, {}
    for k in SUFIJOS:
        base = tarde[k] if tarde[k] > 3 else 0.0
        prev, medidos, primero = base, [], True
        for a, b in zip(bordes, bordes[1:]):
            g0 = max(a + 0.3, prev + 0.5) if base else a + 0.3
            ancho = 120.0 if primero else 60.0
            g1 = min(b - 0.3, g0 + ancho)
            if g1 - g0 < 1.5:
                medidos.append({"desde": round(a, 3), "s": None if primero and base else prev, "c": None})
                continue
            # el primero de un archivo que arranco tarde: la hora del nombre es de reloj y las pausas previas la
            # corren, asi que se busca en +-30 s
            maximo = 30.0 if primero and base and k not in del_log else 3.0
            o0 = max(0.0, g0 - prev)                    # el mismo instante segun el desfase anterior
            d, c = desfase(mono16k(guia, g1 - g0, g0), mono16k(archivos[k], g1 - g0 + maximo, o0), maximo=maximo)
            d += g0 - o0
            if primero and c < 0.5:
                sys.exit(f"el audio de {archivos[k].name} no se parece al de la guia (correlacion {c:.2f}): "
                         "revisa que el filtro grabe la pista 1 (o All)")
            if c < 0.5:
                print(f"    aviso: {k} {a:.1f}-{b:.1f}s correlacion {c:.2f}: se queda el desfase anterior")
                d = prev
            medidos.append({"desde": round(a, 3), "s": round(d, 4), "c": round(c, 2)})
            prev, primero = d, False
        # un tramo sin archivo todavia (arranco tarde) queda en None; los cortos heredan el anterior
        por_tipo[k] = medidos
        valores = [m["s"] for m in medidos if m["s"] is not None]
        desf[k] = min(valores)
        dist = sorted(set(valores))
        print(f"  desfase {k}: {', '.join(f'{v * 1000:+.1f}' for v in dist)} ms en {len(medidos)} tramos")
        if base:
            sin = [m["desde"] for m in medidos if m["s"] is None]
            print(f"    sin imagen de {k} en los tramos que empiezan en {sin} s (arranco tarde)")
    # tramos combinados: [{desde, camara: [clave, desfase] | None, pantalla: [...]}], sin repetir iguales
    tramos = []
    for j, a in enumerate(bordes[:-1]):
        t = {"desde": round(a, 3), **{k: ([k, por_tipo[k][j]["s"]] if por_tipo[k][j]["s"] is not None else None)
                                     for k in SUFIJOS}}
        if not tramos or any(tramos[-1][k] != t[k] for k in SUFIJOS):
            tramos.append(t)

    escena_pip = next((e for e, v in mapa_esc.items() if v == "pantalla_camara"), None)
    pip = pip_de_obs(escena_pip) if escena_pip else dict(PIP_DEFECTO, de="medido 2026-10-23")
    print(f"  circulo: centro ({pip['cx']}, {pip['cy']}), {pip['d']} px; camara {pip['recorte']} ({pip['de']})")

    f = {"guia": str(guia).replace("\\", "/"),
         "archivos": {k: str(p).replace("\\", "/") for k, p in archivos.items() if k != "guia"},
         "desfase": desf, "tramos": tramos, "medidos": por_tipo, "pausas": pausas, "duracion": dur_g,
         "escenas": escenas, "vistas_por_escena": mapa_esc, "pip": pip,
         "log": "obs_log.txt" if log else None, "generado": datetime.now().isoformat(timespec="seconds")}
    (edit / "fuentes.json").write_text(json.dumps(f, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {edit / 'fuentes.json'}")


def juntar(sesiones: list[Path], edit: Path) -> None:
    """Varias grabaciones de un mismo video (cada una ya con su `ingesta`) en UNA linea de tiempo: la guia de
    todas seguidas (guia_juntas.mov, la unica que se recodifica: 720p) y un fuentes.json cuyos tramos apuntan a
    la camara y la pantalla de cada sesion, que se quedan como estan. Cada sesion entra con un largo exacto en
    cuadros (el audio se recorta al cuadro; la cola de una grabacion es silencio), asi el audio de la sesion N
    empieza justo en la suma de las anteriores y los desfases se trasladan sin error."""
    fs = [json.loads((s / "fuentes.json").read_text(encoding="utf-8")) for s in sesiones]
    edit.mkdir(parents=True, exist_ok=True)
    largos = [int(duracion(Path(f["guia"])) * FPS) / FPS for f in fs]
    inicio = [round(sum(largos[:i]), 4) for i in range(len(fs))]

    dst = edit / "guia_juntas.mov"
    n = len(fs)
    ins, fc, cat = [], [], ""
    for i, (f, d) in enumerate(zip(fs, largos)):
        ins += ["-i", f["guia"]]
        fc.append(f"[{i}:v:0]fps={FPS},tpad=stop_mode=clone:stop_duration=2,trim=duration={d:.4f},"
                  f"setpts=PTS-STARTPTS[v{i}]")
        for a in (0, 1):
            fc.append(f"[{i}:a:{a}]aresample=48000,apad,atrim=duration={d:.4f},asetpts=PTS-STARTPTS[a{i}_{a}]")
        cat += f"[v{i}][a{i}_0][a{i}_1]"
    fc.append(f"{cat}concat=n={n}:v=1:a=2[v][a0][a1]")
    if not dst.exists():
        print(f"  {dst.name}: {n} sesiones, {sum(largos) / 60:.1f} min (recodifica la guia de 720p)", flush=True)
        tmp = dst.with_suffix(".tmp.mov")
        subprocess.run(["ffmpeg", "-v", "error", "-stats", "-y", *ins, "-filter_complex", ";".join(fc),
                        "-map", "[v]", "-map", "[a0]", "-map", "[a1]", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", str(tmp)], check=True)
        tmp.replace(dst)
    real = duracion(dst)
    if abs(real - sum(largos)) > 0.01:
        sys.exit(f"{dst.name} dura {real:.3f} s y deberia durar {sum(largos):.3f} s")
    wav = edit / "mic16k.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(dst), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(wav)], check=True)

    archivos, tramos, escenas, mapa, pausas, sesion = {}, [], [], {}, [], []
    for i, (f, g0, d) in enumerate(zip(fs, inicio, largos), start=1):
        for k, p in f["archivos"].items():
            archivos[f"{k}_s{i}"] = p
        for tr in f["tramos"]:
            if tr["desde"] < d:
                tramos.append({"desde": round(g0 + tr["desde"], 4),
                               **{k: ([f"{v[0]}_s{i}", round(g0 + v[1], 4)] if v else None)
                                  for k, v in tr.items() if k != "desde"}})
        for e in f["escenas"]:
            if e["desde"] < d:
                escenas.append(dict(e, desde=round(g0 + e["desde"], 3)))
        mapa.update(f["vistas_por_escena"])
        pausas += [round(g0 + p, 3) for p in f.get("pausas", []) if p < d]
        sesion.append({"n": i, "desde": g0, "hasta": round(g0 + d, 4), "guia": f["guia"],
                       "edit": str(sesiones[i - 1]).replace("\\", "/")})
        print(f"  sesion {i}: {g0:8.2f} - {g0 + d:8.2f} s  ({Path(f['guia']).name})")
    out = {"guia": str(dst).replace("\\", "/"), "archivos": archivos,
           "desfase": {k: tramos[0][k][1] for k in SUFIJOS if tramos[0].get(k)}, "tramos": tramos,
           "pausas": pausas, "duracion": round(sum(largos), 4), "sesiones": sesion, "escenas": escenas,
           "vistas_por_escena": mapa, "pip": fs[0]["pip"], "generado": datetime.now().isoformat(timespec="seconds")}
    (edit / "fuentes.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {len(tramos)} tramos, {len(escenas)} escenas -> {edit / 'fuentes.json'}")
    print(f"  (transcribir mic16k.wav con -o transcripts/guia.json y build_cleanup_edl sobre {dst.name} --key guia)")


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
    # tambien se parte donde cambia el archivo o el desfase (pausas que corren el Source Record, otra sesion)
    cambios = sorted({e["desde"] for e in escenas[1:]} | {t["desde"] for t in (f.get("tramos") or [])[1:]})

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

    # un tramo sin el archivo que pide su vista (la pantalla arranco tarde): la imagen sale de la guia (720p)
    sin_archivo = 0
    if f.get("tramos"):
        for p in nuevos:
            if p.get("source") != kg or not p.get("vista"):
                continue
            t = p["start"] + 0.01
            if p["vista"] != "camara" and fuente_en(f, "pantalla", t) is None:
                del p["vista"]
                sin_archivo += 1
            elif p["vista"] == "pantalla_camara" and fuente_en(f, "camara", t) is None:
                p["vista"] = "pantalla"
    if sin_archivo:
        print(f"  {sin_archivo} cortes sin archivo de pantalla en su tramo: van con la imagen de la guia")

    bk = edit / "edl_backups"
    bk.mkdir(exist_ok=True)
    shutil.copy2(edl_p, bk / f"edl_antes-vistas_{datetime.now():%Y%m%d-%H%M%S}.json")
    edl["sources"].update(f["archivos"])
    edl["fuentes"] = {"guia": kg, "desfase": f["desfase"], "pip": f["pip"],
                      **({"tramos": f["tramos"]} if f.get("tramos") else {})}
    edl["ranges"] = nuevos
    edl_p.write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
    cuenta = {}
    for r in nuevos:
        cuenta[r.get("vista", "-")] = cuenta.get(r.get("vista", "-"), 0) + 1
    print(f"  {len(nuevos)} cortes ({n_part} partidos en un cambio de escena): "
          + ", ".join(f"{v} {n}" for v, n in sorted(cuenta.items())))


def tramo(origen: Path, desde: float, hasta: float, edit: Path, ancho_cam: int) -> None:
    """Para un short: el pedazo [desde, hasta] (segundos de la guia) de los tres archivos, ya alineados (los
    tres empiezan en el mismo instante, asi el desfase queda en 0), en el edit del short. La camara se baja a
    `ancho_cam` (2560 = margen para reencuadrar el panel de 1080x960 sin agrandar); la pantalla va nativa."""
    fo = json.loads((origen / "fuentes.json").read_text(encoding="utf-8"))
    a = round(round(max(0.0, desde) * FPS) / FPS, 4)
    dur = round(hasta - a, 3)
    edit.mkdir(parents=True, exist_ok=True)
    salidas = {}
    for k, src, extra in (("guia", fo["guia"], []),
                          ("camara", fo["archivos"]["camara"], ["-vf", f"scale={ancho_cam}:-2:flags=lanczos"]),
                          ("pantalla", fo["archivos"]["pantalla"], [])):
        t0 = a - (fo["desfase"].get(k, 0.0) if k != "guia" else 0.0)
        dst = edit / f"fuente_{k}.mov"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.4f}", "-i", src, "-t", f"{dur:.3f}",
                        "-map", "0:v:0", "-map", "0:a?", *extra, "-r", str(FPS), "-c:v", "libx264", "-preset", "slow",
                        "-crf", "14", "-pix_fmt", "yuv420p", "-g", "6", "-keyint_min", "6", "-sc_threshold", "0",
                        "-c:a", "pcm_s16le", "-ar", "48000", str(dst)], check=True)
        salidas[k] = dst
        print(f"  {k:9s} {src.split('/')[-1]} desde {t0:.3f}s -> {dst.name}")
    wav = edit / "mic16k.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(salidas["guia"]), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(wav)], check=True)
    # el desfase de los pedazos se vuelve a medir: deberia salir 0 (si no, la extraccion movio un cuadro)
    g = mono16k(salidas["guia"])
    desf = {}
    for k in SUFIJOS:
        d, c = desfase(g, mono16k(salidas[k]))
        desf[k] = round(d, 4)
        print(f"  desfase del pedazo {k}: {d * 1000:+.1f} ms (correlacion {c:.2f})")
    escenas = [dict(e, desde=round(max(0.0, e["desde"] - a), 3)) for e in fo["escenas"] if e["desde"] < hasta]
    escenas = [e for i, e in enumerate(escenas) if i + 1 == len(escenas) or escenas[i + 1]["desde"] > 0] or \
        [{"desde": 0.0, "escena": "?", "vista": "camara"}]
    escenas[0]["desde"] = 0.0
    k_cam = ancho_cam / probe_ancho(fo["archivos"]["camara"])
    pip = dict(fo["pip"], recorte={c: int(round(v * k_cam)) for c, v in fo["pip"]["recorte"].items()})
    f = {"guia": str(salidas["guia"]).replace("\\", "/"),
         "archivos": {k: str(salidas[k]).replace("\\", "/") for k in SUFIJOS}, "desfase": desf,
         "escenas": escenas, "vistas_por_escena": fo["vistas_por_escena"], "pip": pip,
         "origen": {"edit": str(origen).replace("\\", "/"), "desde": a, "hasta": hasta},
         "generado": datetime.now().isoformat(timespec="seconds")}
    (edit / "fuentes.json").write_text(json.dumps(f, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {edit / 'fuentes.json'}  (transcribir mic16k.wav con la clave 'guia' y build_cleanup_edl sobre "
          f"{salidas['guia'].name})")


def _frame_gris(src: str, t: float) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", src, "-frames:v", "1",
                          "-vf", "scale=160:90,format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).astype(np.float32)


def piezas(origen: Path, spec: Path, edit: Path) -> None:
    """Short desde VARIOS pedazos de un largo con grabacion separada (la forma estandar desde 2026-11-13).
    spec = JSON con [{"nombre", "desde", "hasta", "cara_x"?, "pantalla_zona"?: [x, y, ancho]}] en segundos de la guia del largo,
    en el orden del short. Arma tres fuentes con el MISMO reloj (desfase 0), pegando los pedazos:
      fuente_guia.mov      la guia (720p + mic PCM): transcript, cortes y auditores
      fuente_camara.mov    la camara de cada pedazo, entera y en su 4K nativo (margen para reencuadrar), con el mic
      fuente_pantalla.mov  la pantalla de cada pedazo a 1920x1080 nativa (o la guia si ese tramo no la tiene)
    y escribe el edl.json con los cortes APROBADOS del largo que caen en cada pedazo, ya encuadrados segun la
    escena de OBS: "webcam full" -> camara a cuadro completo; escenas con pantalla -> 50/50 (camara abajo,
    pantalla arriba a 1:1, todo editable en Studio). Tambien transcripts/guia.json y mic16k.wav."""
    edl_o = json.loads((origen / "edl.json").read_text(encoding="utf-8"))
    fo = json.loads((origen / "fuentes.json").read_text(encoding="utf-8"))
    fsep, src_o, kg = edl_o["fuentes"], edl_o["sources"], edl_o["fuentes"]["guia"]
    escenas = [dict(e, vista=fo["vistas_por_escena"].get(e["escena"], e.get("vista", "camara"))) for e in fo["escenas"]]
    words = json.loads((origen / "transcripts" / f"{kg}.json").read_text(encoding="utf-8"))["words"]
    lista = json.loads(spec.read_text(encoding="utf-8"))
    edit.mkdir(parents=True, exist_ok=True)
    tmp = edit / "piezas_tmp"
    tmp.mkdir(exist_ok=True)
    venc = ["-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
            "-g", "6", "-keyint_min", "6", "-sc_threshold", "0"]
    aenc = ["-c:a", "pcm_s16le", "-ar", "48000"]
    rec, cw0 = fo["pip"]["recorte"], 3840
    CW, CH = 3840, 2160                         # camara nativa: el circulo de un programa no aguanta otra reduccion
    segs = {"guia": [], "camara": [], "pantalla": []}
    out_w, ranges, mapa, off, avisos = [], [], {}, 0.0, []
    for p in lista:
        a = round(round(p["desde"] * FPS) / FPS, 4)
        b = round(round(p["hasta"] * FPS) / FPS, 4)
        dur, nom, mid = round(b - a, 4), p["nombre"], (a + b) / 2
        guia = src_o[kg]
        cam = fuente_en(fsep, "camara", mid)
        pan = fuente_en(fsep, "pantalla", mid)
        for k, (src, t0, vf) in {
            "guia": (guia, a, None),
            "camara": (src_o[cam[0]], a - cam[1], f"scale={CW}:{CH}") if cam else (guia, a, f"scale={CW}:{CH}:flags=lanczos"),
            "pantalla": (src_o[pan[0]], a - pan[1], "scale=1920:1080") if pan else (guia, a, "scale=1920:1080:flags=lanczos"),
        }.items():
            dst = tmp / f"{nom}_{k}.mov"
            if not dst.exists():
                # el audio siempre de la guia (el mic): asi la camara suena en Studio igual que la guia
                cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.4f}", "-i", src, "-ss", f"{a:.4f}", "-i", guia,
                       "-t", f"{dur:.4f}", "-map", "0:v:0", "-map", "1:a:0", *(["-map", "1:a:1?"] if k == "guia" else []),
                       *(["-vf", vf] if vf else []), *venc, *aenc, str(dst)]
                subprocess.run(cmd, check=True)
            segs[k].append(dst)
        # escena de OBS a mitad del pedazo y, si es de pantalla, si el archivo "camara" es en realidad el
        # programa (Source Record puesto sobre la escena: sesiones 1-2 de Higgsfield): ahi la camara sola no
        # existe y abajo solo cabe el circulo, a baja resolucion
        vista = vista_en(escenas, mid)
        programa = False
        if cam and vista != "camara":
            dif = np.abs(_frame_gris(src_o[cam[0]], mid - cam[1]) - _frame_gris(guia, mid)).mean()
            programa = bool(dif < 14)
            print(f"  {nom}: escena {vista}, diferencia camara-guia {dif:.1f} -> {'programa' if programa else 'camara sola'}")
        cara_x = p.get("cara_x", (rec["x"] + rec["w"] / 2) * CW / cw0)
        if vista == "camara":
            w = round(CH * 9 / 16)
            enc = {"src_crop": {"x": int(min(max(0, cara_x - w / 2), CW - w)), "y": 0, "w": w, "h": CH},
                   "dst": {"x": 0, "y": 0, "w": 1080, "h": 1920}, "out_size": {"w": 1080, "h": 1920}, "fit": "place"}
            arriba = None
        else:
            if programa:
                k = CW / 1920
                cx, cy, d = fo["pip"]["cx"] * k, fo["pip"]["cy"] * k, fo["pip"]["d"] * k
                h = int(d * 0.66)                   # rectangulo 9:8 inscrito en el circulo (0.664 d): sin sus orillas
                w = int(h * 1080 / 960)
                crop = {"x": int(cx - w / 2), "y": int(cy - h / 2), "w": w, "h": h}
                avisos.append(f"{nom}: la 'camara' de ese tramo es el programa de OBS; abajo va el circulo ampliado "
                              f"{1080 / w:.1f}x (se ve suave)")
            else:
                w = round(CH * 1080 / 960)
                crop = {"x": int(min(max(0, cara_x - w / 2), CW - w)), "y": 0, "w": w, "h": CH}
            enc = {"src_crop": crop, "dst": {"x": 0, "y": 960, "w": 1080, "h": 960},
                   "out_size": {"w": 1080, "h": 1920}, "fit": "place"}
            # zona de la pantalla: [x, y, ancho] en px de la pantalla 1080p (alto 9:8). 1080 de ancho = 1:1; menos
            # ancho agranda el texto (en un celular el texto de una pantalla 1080p a 1:1 casi no se lee)
            px, py, pw_ = p.get("pantalla_zona", (420, 60, 1080))
            arriba = {"src_crop": {"x": px, "y": py, "w": pw_, "h": round(pw_ * 960 / 1080)},
                      "dst": {"x": 0, "y": 0, "w": 1080, "h": 960},
                      "out_size": {"w": 1080, "h": 1920}, "fit": "place"}
        mapa[nom] = {"guia_a": a, "guia_b": b, "offset": round(off, 4), "escena": vista, "programa": programa}
        for w_ in words:
            if a <= w_["start"] and w_["end"] <= b:
                out_w.append(dict(w_, start=round(w_["start"] - a + off, 3), end=round(w_["end"] - a + off, 3)))
        for r in edl_o["ranges"]:
            if r["source"] != kg or r["end"] <= a or r["start"] >= b:
                continue
            s, e = max(r["start"], a), min(r["end"], b)
            n = {"source": "guia", "start": round(s - a + off, 3), "end": round(e - a + off, 3),
                 "beat": r.get("beat", ""), "pieza": nom, "vista": "camara", "reframe": enc}
            if arriba:
                n["arriba"] = arriba
            if r.get("censor"):
                avisos.append(f"{nom}: el corte {r['start']:.2f}s del largo llevaba blur; revisar la pantalla de arriba")
            ranges.append(n)
        off += dur
    for k, files in segs.items():
        lst = tmp / f"lista_{k}.txt"
        lst.write_text("".join(f"file '{f.as_posix()}'\n" for f in files), encoding="utf-8")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-map", "0",
                        "-c", "copy", str(edit / f"fuente_{k}.mov")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(edit / "fuente_guia.mov"), "-map", "0:a:0", "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", str(edit / "mic16k.wav")], check=True)
    (edit / "transcripts").mkdir(exist_ok=True)
    (edit / "transcripts" / "guia.json").write_text(json.dumps(
        {"language_code": "es", "text": " ".join(w["text"] for w in out_w if w.get("type", "word") == "word"),
         "words": out_w}, ensure_ascii=False, indent=1), encoding="utf-8")
    edl = {"sources": {k: str(edit / f"fuente_{k}.mov").replace("\\", "/") for k in segs},
           "fuentes": {"guia": "guia", "desfase": {"camara": 0.0, "pantalla": 0.0}, "pip": fo["pip"]},
           "ranges": ranges, "studio": {"canvas": {"w": 1080, "h": 1920}, "proxy": False, "render": "maestro"},
           "origen": {"largo": str(origen).replace("\\", "/"), "piezas": mapa}}
    edl_p = edit / "edl.json"
    if edl_p.exists():
        bk = edit / "edl_backups"
        bk.mkdir(exist_ok=True)
        shutil.copy2(edl_p, bk / f"edl_antes-piezas_{datetime.now():%Y%m%d-%H%M%S}.json")
    edl_p.write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
    tot = sum(r["end"] - r["start"] for r in ranges)
    print(f"  {len(lista)} pedazos ({off:.1f} s) · {len(ranges)} cortes aprobados del largo · {tot:.1f} s "
          f"({tot / 1.15:.1f} s a 1.15x)")
    for v in avisos:
        print(f"  OJO {v}")


def probe_ancho(src: str) -> int:
    return int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width",
                               "-of", "csv=p=0", src], capture_output=True, text=True, check=True).stdout.strip())


def cincuenta(edit: Path, pantalla_xy: tuple[int, int] | None) -> None:
    """50/50 de los shorts en todos los cortes: la camara abajo (vista camara, un cuadro de 9:8 centrado en la
    cara a resolucion 1:1 del archivo) y la pantalla arriba (una zona de 1080x960 a 1:1: el texto se lee).
    Despues se ajusta todo en Studio."""
    edl_p = edit / "edl.json"
    edl = json.loads(edl_p.read_text(encoding="utf-8"))
    f = edl.get("fuentes")
    if not f:
        sys.exit("el edl.json no tiene fuentes: corre antes `vistas`")
    fuentes = edl["sources"]
    cw, ch = probe_ancho(fuentes["camara"]), None
    ch = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height",
                             "-of", "csv=p=0", fuentes["camara"]], capture_output=True, text=True, check=True).stdout)
    pw = probe_ancho(fuentes["pantalla"])
    # camara: todo el alto, ancho 9:8, centrado en la cara (el centro del recorte del circulo de OBS)
    rc = f["pip"]["recorte"]
    cara_x = rc["x"] + rc["w"] / 2
    w = int(ch * 1080 / 960) // 2 * 2
    x = int(min(max(0, cara_x - w / 2), cw - w))
    cam = {"src_crop": {"x": x, "y": 0, "w": w, "h": ch}, "dst": {"x": 0, "y": 960, "w": 1080, "h": 960},
           "out_size": {"w": 1080, "h": 1920}, "fit": "place"}
    px, py = pantalla_xy or ((pw - 1080) // 2, 60)
    arriba = {"src_crop": {"x": px, "y": py, "w": 1080, "h": 960}, "dst": {"x": 0, "y": 0, "w": 1080, "h": 960},
              "out_size": {"w": 1080, "h": 1920}, "fit": "place"}
    for r in edl["ranges"]:
        r["vista"], r["reframe"], r["arriba"] = "camara", cam, arriba
    edl.setdefault("studio", {}).update({"canvas": {"w": 1080, "h": 1920}, "proxy": False, "render": "maestro"})
    edl_p.write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {len(edl['ranges'])} cortes en 50/50: camara {cam['src_crop']} abajo, pantalla {arriba['src_crop']} arriba")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ingesta", "juntar", "vistas", "tramo", "cincuenta", "piezas"])
    ap.add_argument("--guia", type=Path, help="ingesta: el archivo principal de OBS (la guia de 720p)")
    ap.add_argument("--sesiones", type=Path, nargs="+", help="juntar: los edit de cada ingesta, en orden")
    ap.add_argument("--edit-dir", required=True, type=Path)
    ap.add_argument("--origen", type=Path, help="tramo: el edit del largo (con su fuentes.json)")
    ap.add_argument("--desde", type=float, help="tramo: segundo de la guia donde empieza")
    ap.add_argument("--hasta", type=float, help="tramo: segundo de la guia donde termina")
    ap.add_argument("--ancho-camara", type=int, default=2560, help="tramo: ancho de la camara del pedazo")
    ap.add_argument("--pantalla", help="cincuenta: x,y de la zona de 1080x960 de la pantalla (default centrada)")
    ap.add_argument("--piezas", type=Path, help="piezas: JSON con los pedazos del largo, en el orden del short")
    a = ap.parse_args()
    edit = a.edit_dir.resolve()
    if a.cmd == "ingesta":
        if not a.guia:
            ap.error("ingesta necesita --guia")
        ingesta(a.guia.resolve(), edit)
    elif a.cmd == "juntar":
        if not a.sesiones:
            ap.error("juntar necesita --sesiones")
        juntar([s.resolve() for s in a.sesiones], edit)
    elif a.cmd == "vistas":
        vistas(edit)
    elif a.cmd == "piezas":
        if not (a.origen and a.piezas):
            ap.error("piezas necesita --origen y --piezas")
        piezas(a.origen.resolve(), a.piezas.resolve(), edit)
    elif a.cmd == "tramo":
        if not (a.origen and a.desde is not None and a.hasta is not None):
            ap.error("tramo necesita --origen, --desde y --hasta")
        tramo(a.origen.resolve(), a.desde, a.hasta, edit, a.ancho_camara)
    else:
        xy = tuple(int(v) for v in a.pantalla.split(",")) if a.pantalla else None
        cincuenta(edit, xy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
