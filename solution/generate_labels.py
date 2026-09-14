"""Генерация «меток»: поинстансная оптимизация углов QAOA для h_train.

Каждому из 500 векторов h — свои (gamma, beta); Adam, несколько независимых
случайных инициализаций (рестартов), для каждого инстанса оставляем лучший
рестарт по P(ground). Батч = все 500 инстансов одновременно (симулятор
поддерживает поинстансные углы), поэтому всё векторизуется.

Артефакт: data/labels.npz — gamma (500,5), beta (500,5), p_ground (500).
p_ground здесь — «потолок» (качество полной поинстансной оптимизации) для
каждого инстанса.

Запуск:  python generate_labels.py [--steps 250] [--restarts 3]
Время: ~15-20 мин на CPU (зависит от машины), секунды-минуты на GPU.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA, LABEL_LR, ROOT  # noqa: E402
from QAOA import QAOA  # noqa: E402

P = 5


def run(steps=250, restarts=3, lr=LABEL_LR, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    J = np.load(os.path.join(ROOT, "J.npy"))
    h = np.load(os.path.join(ROOT, "h_train.npy"))
    qaoa = QAOA(J)

    B = len(h)
    ht = torch.tensor(h, dtype=torch.float32)
    ckpt = os.path.join(DATA, "labels.npz")

    best_g = np.zeros((B, P))
    best_b = np.zeros((B, P))
    best_p = np.full(B, -1.0)

    for r in range(restarts):
        g = (torch.rand(B, P) * 0.9 + 0.05).requires_grad_(True)
        b = (torch.rand(B, P) * 0.9 + 0.05).requires_grad_(True)
        opt = torch.optim.Adam([g, b], lr=lr)
        t0 = time.time()
        for step in range(steps):
            opt.zero_grad()
            loss = -qaoa.p_ground(ht, g, b).mean()
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f"  restart {r + 1}/{restarts} step {step}: "
                      f"mean P(ground) = {-loss.item():.4f}", flush=True)

        with torch.no_grad():
            p = qaoa.p_ground(ht, g, b).numpy()
        upd = p > best_p
        if r == 0:
            best_g, best_b = g.detach().numpy().copy(), b.detach().numpy().copy()
        else:
            best_g[upd] = g.detach().numpy()[upd]
            best_b[upd] = b.detach().numpy()[upd]
        best_p = np.where(upd, p, best_p)

        dt = time.time() - t0
        print(f"restart {r + 1}/{restarts}: mean = {p.mean():.4f}, "
              f"улучшилось {int(upd.sum())}/{B}, {dt:.0f} c", flush=True)
        np.savez(ckpt, gamma=best_g, beta=best_b, p_ground=best_p)

    print(f"\nИТОГО по 500 инстансам: mean = {best_p.mean():.4f}, "
          f"median = {np.median(best_p):.4f}, "
          f"min = {best_p.min():.4f}, max = {best_p.max():.4f}", flush=True)
    print(f"сохранено: {ckpt}", flush=True)
    return best_g, best_b, best_p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--restarts", type=int, default=3)
    args = ap.parse_args()
    run(steps=args.steps, restarts=args.restarts)
