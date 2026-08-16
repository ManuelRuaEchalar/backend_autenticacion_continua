#!/usr/bin/env python
"""
Genera un pool de impostores a partir de los datos de OTRO participante real.

POR QUE. Los impostores actuales vienen de HMOG: otros terminales, otro
protocolo (incluia sesiones CAMINANDO) y otra epoca. Medido el 2026-08-15, un
clasificador LINEAL sobre 18 estadisticos basicos separa nuestras ventanas de
las de HMOG con AUC 0.79-0.85, mientras que separar a un usuario real de OTRO
usuario real con la misma app se queda en 0.60. Esa diferencia es brecha de
dominio, y es lo que infla el EER de 1.8% que reporta el modelo.

QUE HACE. Replica exactamente WindowSegmenter sobre la base del par —mismas
sesiones por hueco de 30 s, mismo remuestreo a 50 Hz, mismas ventanas 128/96,
mismo scaler— y aplica el filtro de actividad AUTOCALIBRADO del par (p5 de su
propio suelo de ruido x8, acotado). Emite dos pools DISJUNTOS por sesion, con
los mismos tamanos que los de HMOG para que la comparacion no cambie tambien de
tamano de muestra.

ALCANCE Y PRIVACIDAD. Esto es un artefacto de VALIDACION, no del producto: los
ficheros se empujan a mano por adb y NO van dentro del APK. El sistema
desplegado sigue sin que los datos crudos de un usuario salgan de su
dispositivo. Requiere consentimiento explicito de los participantes para este
analisis concreto.

Uso:
    python generar_pool_par.py <db_del_par> <scaler.json> <dir_salida>
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

WINDOW, STEP, HZ = 128, 96, 50
GAP_MS = 30_000
HISTORY_MS = 14 * 24 * 3600 * 1000

# Mismos tamanos que background_train.bin / background_calib.bin.
N_TRAIN, N_CALIB = 600, 400

# Mismas constantes que WindowSegmenter.filtrarPorActividad.
NOISE_P, K, MIN_T, MAX_T = 5.0, 8.0, 0.01, 0.10


def main() -> None:
    db, scaler_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = json.load(open(scaler_path, encoding="utf-8"))
    mean = np.array(sc["mean"], dtype=np.float64)
    scale = np.array(sc["scale"], dtype=np.float64)

    con = sqlite3.connect(db)
    acc = np.array(list(con.execute(
        "select timestamp,x,y,z from accelerometer_data order by timestamp")),
        dtype=np.float64)
    gyr = np.array(list(con.execute(
        "select timestamp,x,y,z from gyroscope_data order by timestamp")),
        dtype=np.float64)
    now = max(acc[-1, 0], gyr[-1, 0])
    acc = acc[acc[:, 0] >= now - HISTORY_MS]
    at, gt = acc[:, 0], gyr[:, 0]

    # --- sesiones por hueco (detectSessions) ---
    cortes = np.nonzero(np.diff(at) > GAP_MS)[0]
    tramos, ini = [], 0
    for i in cortes + 1:
        if i - 1 > ini:
            tramos.append((ini, i - 1))
        ini = i
    if len(at) - 1 > ini:
        tramos.append((ini, len(at) - 1))

    ventanas, sesion = [], []
    sid = 0
    for a, b in tramos:
        t0, t1 = max(at[a], gt[0]), min(at[b], gt[-1])
        if t1 <= t0:
            continue
        n = int((t1 - t0) / 1000.0 * HZ)
        if n < WINDOW:
            continue
        grid = np.linspace(t0, t1, n)
        ch = np.stack([np.interp(grid, at, acc[:, c]) for c in (1, 2, 3)] +
                      [np.interp(grid, gt, gyr[:, c]) for c in (1, 2, 3)])
        ch = (ch - mean[:, None]) / scale[:, None]
        for s in range(0, n - WINDOW + 1, STEP):
            ventanas.append(ch[:, s:s + WINDOW])
            sesion.append(sid)
        sid += 1

    V = np.array(ventanas)                 # (N, 6, 128)
    S = np.array(sesion)
    print(f"ventanas: {len(V)} de {len(np.unique(S))} sesiones")

    # --- filtro de actividad autocalibrado del PAR ---
    energia = V[:, :3, :].std(axis=2).mean(axis=1)
    suelo = float(np.percentile(energia, NOISE_P))
    umbral = float(np.clip(K * suelo, MIN_T, MAX_T))
    activo = energia >= umbral
    print(f"filtro: suelo(p{NOISE_P})={suelo:.4f} -> umbral={umbral:.4f}; "
          f"{activo.sum()}/{len(V)} activas")
    V, S = V[activo], S[activo]

    if len(V) < N_TRAIN + N_CALIB:
        raise SystemExit(
            f"Solo {len(V)} ventanas activas; hacen falta {N_TRAIN + N_CALIB}."
        )

    # --- particion DISJUNTA POR SESION ---
    # Igual que HMOG reserva sujetos distintos para train y calib: si los dos
    # pools compartieran sesiones, el umbral se calibraria contra el mismo
    # material con el que despues se mide el FAR.
    ses = np.unique(S)
    rng = np.random.RandomState(20260816)
    rng.shuffle(ses)
    corte = max(1, int(len(ses) * 0.6))
    ses_train, ses_calib = set(ses[:corte]), set(ses[corte:])

    def muestrea(conjunto, cuantas, etiqueta):
        idx = np.nonzero([s in conjunto for s in S])[0]
        if len(idx) < cuantas:
            raise SystemExit(
                f"{etiqueta}: solo {len(idx)} ventanas en {len(conjunto)} "
                f"sesiones, hacen falta {cuantas}"
            )
        elegidas = rng.choice(idx, cuantas, replace=False)
        print(f"  {etiqueta}: {cuantas} ventanas de "
              f"{len(np.unique(S[elegidas]))} sesiones")
        return V[np.sort(elegidas)]

    train = muestrea(ses_train, N_TRAIN, "peer_train")
    calib = muestrea(ses_calib, N_CALIB, "peer_calib")

    # (N, 6, 128) -> (N, 128, 6) float32, que es como los lee BackgroundPool.
    for nombre, datos in (("background_peer_train.bin", train),
                          ("background_peer_calib.bin", calib)):
        arr = datos.transpose(0, 2, 1).astype(np.float32)
        ruta = out_dir / nombre
        arr.tofile(ruta)
        print(f"escrito {ruta}  ({arr.shape}, {ruta.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
