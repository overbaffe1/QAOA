# -*- coding: utf-8 -*-
"""make_viz_data.py — сборка viz_data.js для qaoa_viz.html.

Читает:  J.npy, h_train.npy, CSV углов (id,gamma_0..4,beta_0..4, напр. merged2.csv)
Пишет:  viz_data.js  ->  window.VIZ = {J, h, angles, ref_p, ref_mean, tests, meta}

Запуск (в папке с файлами):
    python make_viz_data.py --angles merged2.csv --out viz_data.js

Симулятор в скрипте = тот же FastQAOA (group=4, p=5, 12 кубитов), что и QAOA.py —
numpy, float32-конвейер. Скорость ~20-60 c на 500 инстансов.
"""
import argparse
import json
import time

import numpy as np

P = 5
N = 12
G = 4          # group
DIM = 2 ** N
NG = N // G
D16 = 2 ** G


_gb = np.arange(D16)
_gbits = (_gb[:, None] >> np.arange(G - 1, -1, -1)[None, :]) & 1
K = (_gbits[:, None, :] ^ _gbits[None, :, :]).sum(-1)
PHASE = np.array([1, -1j, -1, 1j], dtype=np.complex128)[K % 4]


def load_angles(path, H):
    d = np.genfromtxt(path, delimiter=",", names=True)
    names = list(d.dtype.names)
    cols = [c for c in names if c != "id"]
    order = [f"gamma_{k}" for k in range(P)] + [f"beta_{k}" for k in range(P)]
    A = np.stack([np.asarray(d[c], dtype=np.float64) for c in order], axis=1)
    ids = np.asarray(d["id"], dtype=np.int64)
    A = A[np.argsort(ids)]          # страховка: строго по id
    if A.shape[0] != H:
        raise SystemExit(f"CSV {path}: {A.shape[0]} строк, ждал {H}")
    return A


def make_data():
    J = np.load("J.npy").astype(np.float64)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    h = np.load("h_train.npy").astype(np.float32)
    H = h.shape[0]
    S = np.zeros((DIM, N), dtype=np.float32)
    for i in range(N):
        S[:, i] = 2 * ((np.arange(DIM) >> (N - 1 - i)) & 1) - 1
    quad = (0.5 * np.einsum("ij,ki,kj->k", J.astype(np.float32), S, S)
            .astype(np.float32))
    E = (quad[None, :] + h @ S.T).astype(np.float32)
    gmin = E.min(axis=1, keepdims=True)
    MASK = (E <= gmin + 1e-9).astype(np.float32)

    return J, h, E, MASK


def p_ground_all(E, MASK, gamma, beta, chunk=50):
    """gamma/beta: (H, P). Возвращает (H,) float64. Чанки, чтобы видно было."""
    H = E.shape[0]
    out = np.empty(H)
    for i0 in range(0, H, chunk):
        i1 = min(i0 + chunk, H)
        psi = np.full((i1 - i0, DIM), 1 / np.sqrt(DIM), dtype=np.complex64)
        for l in range(P):
            psi = psi * (np.exp(1j * (gamma[i0:i1, l][:, None] * E[i0:i1])
                                ).astype(np.complex64))
            c = np.cos(beta[i0:i1, l])[:, None, None]
            s = np.sin(beta[i0:i1, l])[:, None, None]
            M = np.transpose((c ** (G - K) * s ** K).astype(np.complex64) * PHASE,
                             (0, 2, 1))
            for j in range(NG):
                left, right = D16 ** j, D16 ** (NG - 1 - j)
                v = psi.reshape(i1 - i0, left, D16, right).transpose(0, 1, 3, 2)
                v = np.einsum("bni,bij->bnj", v.reshape(i1 - i0, -1, D16), M)
                psi = (v.reshape(i1 - i0, left, right, D16)
                       .transpose(0, 1, 3, 2).reshape(i1 - i0, DIM)
                       .astype(np.complex64))
        prob = (psi.real ** 2 + psi.imag ** 2).astype(np.float32)
        out[i0:i1] = (prob * MASK[i0:i1]).sum(axis=1)
        print(f"  P(ground): {i1}/{H}  mean_so_far={out[:i1].mean():.4f}",
              flush=True)
    return out


def make_tests(h, E, MASK):
    """2 эталонных вектора (считаются numpy-эталоном в момент сборки)."""
    rng = np.random.default_rng(12345)
    tests = []
    for i in (0, 311):
        gg = rng.uniform(-2, 3, (1, P)).astype(np.float32)
        bb = rng.uniform(0, 3.14, (1, P)).astype(np.float32)
        pv = p_ground_all(E[i:i + 1], MASK[i:i + 1], gg, bb)[0]
        tests.append({"h": h[i].tolist(), "g": gg[0].tolist(),
                      "b": bb[0].tolist(), "P": float(pv)})
    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", default="merged2.csv",
                    help="CSV текущих лучших углов (id,gamma_*,beta_*)")
    ap.add_argument("--out", default="viz_data.js")
    args = ap.parse_args()

    t0 = time.time()
    J, h, E, MASK = make_data()
    H = h.shape[0]
    print(f"J {J.shape}, h {h.shape}; ref = {args.angles}")
    A = load_angles(args.angles, H)
    print(f"angles: {A.shape}; mean ref P (numpy):")
    t1 = time.time()
    ref_p = p_ground_all(E, MASK, A[:, :P], A[:, P:])
    ref_mean = float(ref_p.mean())
    print(f"done in {time.time() - t1:.1f} s")

    tests = make_tests(h, E, MASK)
    viz = {
        "J": J.tolist(),
        "h": h.tolist(),
        "angles": A.tolist(),
        "ref_p": ref_p.tolist(),
        "ref_mean": ref_mean,
        "tests": tests,
        "meta": {"n": N, "p": P, "group": G, "dim": DIM,
                 "H": H, "src_angles": args.angles,
                 "made": int(time.time())},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("window.VIZ = ")
        f.write(json.dumps(viz, separators=(",", ":")))
        f.write(";\n")
    import os
    sz = os.path.getsize(args.out)
    print(f"-> {args.out}  ({sz/1024:.0f} KB)  |  ref_mean={ref_mean:.6f} "
          f"(>0.5: {int((ref_p > 0.5).sum())}/{H})  |  total {time.time()-t0:.0f} s")
    print("теперь открой qaoa_viz.html в браузере (viz_data.js рядом).")


if __name__ == "__main__":
    main()