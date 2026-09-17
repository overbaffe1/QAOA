"""KNN-WARM-START: посылка для h_test из твоих лучших углов (merged.csv).

Идея (проверена конкурентом на LOO: 0.3163 vs оракул 0.3175, потеря 0.4%):
для каждого h_test взять K ближайших h_train по знаковой метрике
min(‖h−h'‖, ‖h+h'‖)  (симметрия P(h)=P(−h)), взять ИХ углы из merged.csv,
сделать по Jitter джиттер-копий (σ), дожать Adam по P_ground, взять лучшее.
~5-10 мин на Colab T4 (FastQAOA), влезает в 10-минутный бюджет.

Запуск:
  python knn_warm.py --h h_test.npy --targets merged.csv --out submission_knn.csv
  # проверка на train (LOO-аналог): --h h_train.npy (число должно быть ~0.31+)
"""
import argparse
import os
import time

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from QAOA import FastQAOA, P  # noqa: E402


def load_targets(path):
    ids, A = [], []
    with open(path) as f:
        next(f)  # header
        for line in f:
            row = line.strip().split(",")
            ids.append(int(float(row[0])))
            A.append([float(x) for x in row[1:1 + P * 2]])
    return np.array(ids), np.array(A, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="h_test.npy", help="инференс-матрица (h_test или h_train)")
    ap.add_argument("--h-train", default="h_train.npy")
    ap.add_argument("--targets", default="merged.csv",
                    help="лучшие углы по h_train (id + 10 углов)")
    ap.add_argument("--j", default="J.npy")
    ap.add_argument("--out", default="submission_knn.csv")
    ap.add_argument("--k", type=int, default=8, help="соседей h_train")
    ap.add_argument("--jitter", type=int, default=4, help="джиттер-копий на соседа")
    ap.add_argument("--sigma", type=float, default=0.08, help="σ джиттера")
    ap.add_argument("--steps", type=int, default=100, help="Adam-шагов")
    ap.add_argument("--chunk", type=int, default=16,
                    help="инстансов за батч Adam (VRAM-баланс)")
    ap.add_argument("--time-budget", type=float, default=540.0,
                    help="сек на Adam-дожим; после — выгружаем лучшее без дожима")
    args = ap.parse_args()

    t0 = time.time()
    h_test = np.load(args.h).astype(np.float32)
    h_train = np.load(args.h_train).astype(np.float32)
    ids_tr, A_tr = load_targets(args.targets)
    # сходим по id train
    order = np.argsort(ids_tr)
    h_train = h_train[order]
    A_tr = A_tr[order]
    T = h_test.shape[0]
    print(f"инференс: {T} h | соседи K={args.k} x jitter {args.jitter} | "
          f"Adam {args.steps} | device={DEVICE}", flush=True)

    # знаковая метрика min(‖h−h'‖, ‖h+h'‖)
    D = np.linalg.norm(h_test[:, None, :] - h_train[None, :, :], axis=2)
    Dp = np.linalg.norm(h_test[:, None, :] + h_train[None, :, :], axis=2)
    nn = np.argsort(np.minimum(D, Dp), axis=1)[:, :args.k]  # (T, K)

    qaoa = FastQAOA(np.load(args.j), device=DEVICE)
    ht_all = torch.tensor(h_test, dtype=torch.float32, device=DEVICE)

    rng = np.random.default_rng(12345)
    NJ = args.k * args.jitter
    best_a = A_tr[nn[:, 0]].copy()   # стартовый best = первый сосед
    best_p = np.zeros(T, dtype=np.float32)

    def eval_block(ib0, a_block):
        g = torch.tensor(a_block[:, :P], dtype=torch.float32, device=DEVICE)
        b = torch.tensor(a_block[:, P:], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return qaoa.p_ground(ht_all[ib0:ib0 + len(a_block)], g, b).cpu().numpy()

    best_p = eval_block(0, best_a)

    for ib0 in range(0, T, args.chunk):
        n_i = min(args.chunk, T - ib0)
        # кандидаты: n_i x NJ строк (соседей x джиттер), первый джиттер = чистые углы
        cand = np.empty((n_i * NJ, P * 2), dtype=np.float32)
        for i in range(n_i):
            for t in range(args.k):
                base = A_tr[nn[ib0 + i, t]]
                for j in range(args.jitter):
                    r0 = i * NJ + t * args.jitter + j
                    cand[r0] = base if j == 0 else base + rng.normal(
                        0.0, args.sigma, P * 2)
        g = torch.tensor(cand[:, :P], dtype=torch.float32,
                         device=DEVICE, requires_grad=True)
        b = torch.tensor(cand[:, P:], dtype=torch.float32,
                         device=DEVICE, requires_grad=True)
        opt = torch.optim.Adam([g, b], lr=0.05)
        ht_c = ht_all[ib0:ib0 + n_i].repeat_interleave(NJ, dim=0)
        for s in range(args.steps):
            if time.time() - t0 > args.time_budget:
                break
            opt.zero_grad()
            loss = -qaoa.p_ground(ht_c, g, b).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p_c = qaoa.p_ground(ht_c, g, b).cpu().numpy().reshape(n_i, NJ)
        a_c = np.concatenate([g.detach().cpu().numpy(),
                              b.detach().cpu().numpy()], axis=1)
        am = p_c.argmax(axis=1)
        # ratchet: только лучше текущего best
        upd = p_c.max(axis=1) > best_p[ib0:ib0 + n_i]
        for i in range(n_i):
            if upd[i]:
                best_a[ib0 + i] = a_c[i * NJ + am[i]]
                best_p[ib0 + i] = p_c[i, am[i]]
        done = ib0 + n_i
        print(f"  {done}/{T}: running mean P = {best_p[:done].mean():.4f} "
              f"({time.time() - t0:.0f} c)", flush=True)

    cols = ["id"] + [f"gamma_{k}" for k in range(P)] + [f"beta_{k}" for k in range(P)]
    out = np.concatenate([np.arange(T)[:, None], best_a], axis=1)
    np.savetxt(args.out, out, delimiter=",", header=",".join(cols),
               comments="", fmt=["%d"] + ["%.12f"] * (P * 2))
    print(f"ГОТОВО: mean P = {best_p.mean():.4f} (max {best_p.max():.3f}) "
          f"-> {args.out} за {time.time() - t0:.0f} c", flush=True)


if __name__ == "__main__":
    main()
