#!/usr/bin/env python3
"""Voz procesada con Auphonic (experimento, 2026-11-04): solo los tramos que sobreviven al corte.

No se sube la grabacion cruda (en el largo del CRM eran 24.8 min para 7 de video): se toman los
ranges del edl.json con un margen, se juntan en "islas" y solo esas se suben con el preset de
Auphonic (por defecto "Studio Voice medio"). Lo que vuelve se alinea contra lo subido (quita el
jingle del plan gratis y el retraso de Studio Voice) y se recoloca en la linea de tiempo de la guia:
<edit>/voz/voz_auphonic.wav. Con "voz" en el edl.json, render.py (largos) y build_final (shorts)
toman la voz de ahi; si un corte cae fuera de las islas se detienen en vez de mezclar voz cruda.

Correrlo despues de from-studio y antes de build_final:

    auphonic_voz.py --edit-dir E [--preset "Studio Voice medio"] [--margen 1.0] [--dry-run]

Tras un cambio de cortes procesa solo las islas que faltan, con islas vecinas como contexto
(Auphonic cobra minimo 3 min) para igualar el nivel con lo ya procesado. Quitar "voz" del edl.json
vuelve a la voz cruda. La API key sale de la variable de entorno AUPHONIC_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

SR = 48000
HOP = SR // 1000                 # envolvente a 1 ms
PAD = 2.0                        # silencio a cada lado de lo subido: un fundido del jingle cae ahi
ESPACIO = 0.25                   # silencio entre islas: cada union es una pausa, no un empalme
JUNTAR = 1.0                     # islas separadas por menos de esto se suben como una sola
MIN_COBRO = 180.0                # Auphonic cobra minimo 3 min por produccion
DESFASE_MAX = 20.0               # jingle + retraso: el inicio de lo subido aparece antes de 20 s
DERIVA_MAX_MS = 20               # entre inicio y final; mas que eso no es un desfase fijo
CORR_MIN = 0.9
API = "https://auphonic.com/api"


# --- audio ------------------------------------------------------------------------------------

def leer_pcm(path: Path) -> np.ndarray:
    """Pista 0:a:0 como int16 (n, 2) a 48 kHz."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "2", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).reshape(-1, 2).copy()


def escribir_wav(path: Path, x: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.wav")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.ascontiguousarray(x, dtype=np.int16).tobytes())
    tmp.replace(path)


def escribir_flac(path: Path, x: np.ndarray) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "-",
                    "-c:a", "flac", str(path)], input=np.ascontiguousarray(x, dtype=np.int16).tobytes(), check=True)


def envolvente(x: np.ndarray) -> np.ndarray:
    """Energia en dB por milisegundo (mono). Sirve aunque Studio Voice regenere la forma de onda."""
    m = x.astype(np.float32).mean(axis=1) / 32768.0
    n = len(m) // HOP
    e = np.sqrt((m[:n * HOP].reshape(n, HOP) ** 2).mean(axis=1))
    return np.log10(e + 1e-5).astype(np.float64)


def mejor_desfase(ref: np.ndarray, sig: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    """Posicion k (ms) en [lo, hi] donde ref mejor coincide con sig[k:k+len(ref)], y su correlacion."""
    ref = ref - ref.mean()
    L = len(ref)
    hi = min(hi, len(sig) - L)
    if hi < lo or np.linalg.norm(ref) < 1e-6:
        raise SystemExit("no hay con que alinear (ventana vacia o sin voz)")
    seg = sig[lo:hi + L]
    nfft = 1 << (len(seg) + L).bit_length()
    c = np.fft.irfft(np.fft.rfft(seg, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)[:hi - lo + 1]
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    cs2 = np.concatenate([[0.0], np.cumsum(seg ** 2)])
    s1, s2 = cs[L:] - cs[:-L], cs2[L:] - cs2[:-L]
    var = np.maximum(s2 - s1 ** 2 / L, 1e-12)
    r = c / (np.linalg.norm(ref) * np.sqrt(var))
    k = int(np.argmax(r))
    return lo + k, float(r[k])


def db_por_trama(x: np.ndarray, n: int = SR // 10) -> np.ndarray:
    m = x.astype(np.float32).mean(axis=1) / 32768.0
    k = len(m) // n
    return 20 * np.log10(np.sqrt((m[:k * n].reshape(k, n) ** 2).mean(axis=1)) + 1e-9)


# --- islas ------------------------------------------------------------------------------------

def juntar(iv: list[tuple[float, float]], hueco: float = 0.0) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + hueco:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def contenida(a: float, b: float, islas: list) -> bool:
    return any(x - 1e-3 <= a and b <= y + 1e-3 for x, y in islas)


def plan(necesarias: list, cubiertas: list) -> tuple[list, list]:
    """(islas a escribir, islas vecinas que solo van de contexto para igualar el nivel)."""
    faltan = [n for n in necesarias if not contenida(n[0], n[1], cubiertas)]
    if not cubiertas:
        return faltan, []
    # una isla que falta absorbe las cubiertas que toca: las uniones vuelven a caer en un margen
    nuevas = juntar([tuple(f) for f in faltan] + [tuple(c) for c in cubiertas
                                                  if any(c[0] <= f[1] and f[0] <= c[1] for f in faltan)])
    nuevas = [n for n in nuevas if not contenida(n[0], n[1], cubiertas)]
    vecinas = [c for c in cubiertas if not any(c[0] <= n[1] and n[0] <= c[1] for n in nuevas)]
    vecinas.sort(key=lambda c: min(max(0.0, c[0] - n[1], n[0] - c[1]) for n in nuevas) if nuevas else 0)
    total = sum(b - a for a, b in nuevas)
    contexto = []
    for c in vecinas:                      # gratis hasta el minimo de cobro, y bastan unos 30 s de voz
        if total >= MIN_COBRO and sum(b - a for a, b in contexto) >= 30:
            break
        contexto.append(c)
        total += c[1] - c[0]
    return nuevas, sorted(contexto)


# --- API --------------------------------------------------------------------------------------

def _req():
    import requests
    key = os.environ.get("AUPHONIC_API_KEY")
    if not key:
        raise SystemExit("falta la variable de entorno AUPHONIC_API_KEY (y reiniciar Claude Code despues de crearla)")
    return requests, {"Authorization": f"bearer {key}"}


def procesar_en_auphonic(entrada: Path, preset: str, titulo: str, salida: Path) -> dict:
    """Sube, espera y baja el WAV. Devuelve los datos de la produccion."""
    requests, h = _req()
    print(f"  subiendo {entrada.name} ({entrada.stat().st_size / 1e6:.0f} MB) con el preset '{preset}'", flush=True)
    with open(entrada, "rb") as f:
        r = requests.post(f"{API}/simple/productions.json", headers=h, timeout=1800,
                          data={"preset": preset, "title": titulo, "action": "start"}, files={"input_file": f})
    r.raise_for_status()
    if not (r.json().get("data") or {}).get("uuid"):
        raise SystemExit(f"Auphonic no creo la produccion: {r.json().get('error_message') or r.text[:300]}")
    uuid = r.json()["data"]["uuid"]
    print(f"  produccion {uuid}: https://auphonic.com/engine/status/{uuid}", flush=True)
    t0, ultimo = time.time(), None
    while True:
        d = requests.get(f"{API}/production/{uuid}.json", headers=h, timeout=60).json()["data"]
        if d.get("status_string") != ultimo:
            ultimo = d.get("status_string")
            print(f"    {ultimo} ({time.time() - t0:.0f} s)", flush=True)
        if d.get("error_status") or ultimo == "Error":
            raise SystemExit(f"Auphonic fallo: {d.get('error_message') or ultimo}")
        if d.get("status") == 3:
            break
        if time.time() - t0 > 5400:
            raise SystemExit(f"Auphonic no termino en 90 min; revisar {uuid}")
        time.sleep(10)
    wavs = [o for o in d.get("output_files") or [] if o.get("format") == "wav"]
    if not wavs:
        raise SystemExit(f"el preset '{preset}' no tiene salida WAV 16-bit: agregala en Auphonic")
    with requests.get(wavs[0]["download_url"], headers=h, stream=True, timeout=1800) as r:
        r.raise_for_status()
        with open(salida, "wb") as f:
            for trozo in r.iter_content(1 << 20):
                f.write(trozo)
    return d


# --- principal --------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Voz procesada con Auphonic, solo en los tramos del corte")
    ap.add_argument("--edit-dir", type=Path, required=True)
    ap.add_argument("--preset", default="Studio Voice medio")
    ap.add_argument("--margen", type=float, default=1.0, help="s alrededor de cada corte (contexto para el modelo)")
    ap.add_argument("--fuente", help="clave de la fuente de la voz (por defecto la guia de la grabacion separada)")
    ap.add_argument("--dry-run", action="store_true", help="solo dice que subiria y cuantos minutos cobra")
    args = ap.parse_args(argv)

    edit = args.edit_dir.resolve()
    edl_p = edit / "edl.json"
    edl = json.loads(edl_p.read_text(encoding="utf-8"))
    voz = edl.get("voz") or {}
    fuente = args.fuente or voz.get("fuente") or (edl.get("fuentes") or {}).get("guia")
    if not fuente:
        claves = sorted({r["source"] for r in edl["ranges"]})
        if len(claves) != 1:
            raise SystemExit(f"varias fuentes en el EDL ({claves}): indica --fuente")
        fuente = claves[0]
    src = Path(edl["sources"][fuente])
    src = src if src.is_absolute() else (edit / src).resolve()
    if voz.get("preset") and voz["preset"] != args.preset:
        raise SystemExit(f"la voz ya procesada usa '{voz['preset']}': mezclar presets dejaria tramos distintos. "
                         f"Borra 'voz' del edl.json y edit/voz/ para empezar de cero")
    if edl.get("audio_filter"):
        print(f"  aviso: el EDL trae audio_filter ({edl['audio_filter']}); se aplicaria encima de la voz procesada")

    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(src)], capture_output=True, text=True, check=True).stdout.strip())
    rs = [r for r in edl["ranges"] if r["source"] == fuente]
    necesarias = juntar([(max(0.0, r["start"] - args.margen), min(dur, r["end"] + args.margen)) for r in rs], JUNTAR)
    cubiertas = voz.get("islas") or []
    nuevas, contexto = plan(necesarias, cubiertas)
    if not nuevas:
        print(f"voz: los {len(rs)} cortes ya caen en la voz procesada; nada que subir")
        return
    subir = sorted([(a, b, True) for a, b in nuevas] + [(a, b, False) for a, b in contexto])   # True = se escribe
    seg = sum(b - a for a, b, _ in subir) + ESPACIO * (len(subir) - 1) + 2 * PAD
    print(f"voz: {len(nuevas)} islas nuevas ({sum(b - a for a, b in nuevas) / 60:.1f} min)"
          + (f" + {len(contexto)} vecinas de contexto" if contexto else "")
          + f" -> se suben {seg / 60:.1f} min (cobra {max(seg, MIN_COBRO) / 60:.1f} min, mas el jingle si es gratis)")
    if args.dry_run:
        return

    print(f"  leyendo la voz de {src.name}", flush=True)
    crudo = leer_pcm(src)
    dvoz = edit / "voz"
    dvoz.mkdir(exist_ok=True)
    archivo = dvoz / "voz_auphonic.wav"
    linea = leer_pcm(archivo) if archivo.exists() and cubiertas else crudo.copy()
    if len(linea) != len(crudo):
        raise SystemExit(f"{archivo.name} no mide lo mismo que la guia: borralo junto con 'voz' del edl.json")

    # lo que se sube: PAD + isla + ESPACIO + isla ... + PAD, y donde cae cada isla
    pad, esp = np.zeros((int(PAD * SR), 2), np.int16), np.zeros((int(ESPACIO * SR), 2), np.int16)
    partes, pos, n = [pad], [], len(pad)
    for a, b, nueva in subir:
        i0, i1 = int(round(a * SR)), int(round(b * SR))
        if pos:
            partes.append(esp)
            n += len(esp)
        pos.append((n, i0, i1, nueva))
        partes.append(crudo[i0:i1])
        n += i1 - i0
    partes.append(pad)
    subida = np.concatenate(partes)

    with tempfile.TemporaryDirectory(dir=dvoz) as tmp:
        ent, sal = Path(tmp) / "islas.flac", Path(tmp) / "auphonic.wav"
        escribir_flac(ent, subida)
        datos = procesar_en_auphonic(ent, args.preset, f"{edit.parent.name} voz {datetime.now():%Y-%m-%d %H:%M}", sal)
        proc = leer_pcm(sal)

    # alinear: desfase al inicio y al final de lo subido (jingle + retraso del modelo), lineal entre los dos
    eu, ep = envolvente(subida), envolvente(proc)
    c0, c1 = int(PAD * 1000), len(eu) - int(PAD * 1000)
    w = min(60000, (c1 - c0) // 2)
    p0, r0 = mejor_desfase(eu[c0:c0 + w], ep, c0, c0 + int(DESFASE_MAX * 1000))
    p1, r1 = mejor_desfase(eu[c1 - w:c1], ep, c1 - w, c1 - w + int(DESFASE_MAX * 1000))
    d0, d1 = p0 - c0, p1 - (c1 - w)
    print(f"  desfase {d0} ms al inicio y {d1} ms al final (correlacion {r0:.2f} / {r1:.2f})")
    if min(r0, r1) < CORR_MIN or abs(d1 - d0) > DERIVA_MAX_MS:
        raise SystemExit("la salida de Auphonic no se alinea con lo subido (corto algo o cambio el tiempo): "
                         "revisa que el preset no tenga cortes; la voz queda como estaba")
    t0, t1 = (c0 + w / 2) / 1000, (c1 - w / 2) / 1000

    def tramo_proc(u0: int, largo: int) -> np.ndarray:
        d = d0 + (d1 - d0) * (u0 / SR - t0) / (t1 - t0)
        k = u0 + int(round(d * HOP))
        x = proc[k:k + largo]
        if len(x) < largo:
            raise SystemExit("la salida de Auphonic es mas corta que lo subido")
        return x

    # nivel: en una corrida incremental, las vecinas comparan lo nuevo contra lo ya procesado
    g = 0.0
    if contexto:
        dif = []
        for u0, i0, i1, nueva in pos:
            if not nueva:
                viejo, nuevo = db_por_trama(linea[i0:i1]), db_por_trama(tramo_proc(u0, i1 - i0))
                m = (viejo > -45) & (nuevo > -45)
                dif += list((viejo - nuevo)[m])
        if len(dif) >= 20:
            g = float(np.median(dif))
            print(f"  nivel: lo nuevo se ajusta {g:+.2f} dB para igualar lo ya procesado ({len(dif)} tramas)")
            if abs(g) > 6:
                print("  aviso: diferencia de nivel grande entre producciones; escuchar las uniones")
    k = 10 ** (g / 20)
    for u0, i0, i1, nueva in pos:
        if nueva:
            x = tramo_proc(u0, i1 - i0).astype(np.float32) * k
            linea[i0:i1] = np.clip(np.round(x), -32768, 32767).astype(np.int16)
    escribir_wav(archivo, linea)

    bk = edit / "edl_backups"
    bk.mkdir(exist_ok=True)
    shutil.copy2(edl_p, bk / f"edl_antes-voz_{datetime.now():%Y%m%d-%H%M%S}.json")
    edl = json.loads(edl_p.read_text(encoding="utf-8"))          # por si algo lo cambio mientras Auphonic procesaba
    voz = edl.get("voz") or {}
    voz.update({"fuente": fuente, "archivo": str(archivo).replace("\\", "/"), "preset": args.preset,
                "islas": juntar([tuple(c) for c in cubiertas] + [tuple(x) for x in nuevas])})
    voz.setdefault("producciones", []).append({"uuid": datos.get("uuid"), "fecha": f"{datetime.now():%Y-%m-%d %H:%M}",
                                               "min_subidos": round(seg / 60, 2),
                                               "creditos_h": (datos.get("used_credits") or {}).get("combined")})
    edl["voz"] = voz
    edl_p.write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
    faltan = [r for r in rs if not contenida(r["start"], r["end"], voz["islas"])]
    print(f"voz: {archivo}  ({len(voz['islas'])} islas; {len(rs) - len(faltan)}/{len(rs)} cortes cubiertos)")
    if faltan:
        raise SystemExit(f"{len(faltan)} cortes siguen fuera de la voz procesada")


if __name__ == "__main__":
    main()
