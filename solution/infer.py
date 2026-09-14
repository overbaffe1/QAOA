"""Инференс: модель -> углы -> короткая «полировка» -> submission.csv.

Этапы:
  1. Сеть f(h) -> (gamma, beta)  (миллисекунды на все 500);
  2. Полировка: от предсказания сети — POLISH_STEPS шагов Adam поинстансно
     (каждый инстанс дорабатывает свои углы). Это финальная доводка к
     локальному оптимуму P(ground) конкретного h: тысячи запусков схемы
     заменяются десятками, при этом теряются доли процента P(ground)
     против полной оптимизации.
  3. Запись submission.csv (id, gamma_0..gamma_4, beta_0..beta_4).

Время: на CPU ~2-3 мин на 500 инстансов (лимит 10 мин), на GPU — секунды.

Запуск:  python infer.py [--h h_test.npy] [--out submission.csv]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA, POLISH_LR, POLISH_STEPS, ROOT  # noqa: E402
from features import build_features  # noqa: E402
from model import QAOAAngleNet  # noqa: E402
from QAOA import QAOA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="h_test.npy", help="файл с h (по умолчанию h_test.npy)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--polish", type=int, default=POLISH_STEPS)
    args = ap.parse_args()

    h_path = args.h if os.path.exists(args.h) else os.path.join(ROOT, args.h)
    if not os.path.exists(h_path):
        sys.exit(f"файл {h_path} не найден")
    h = np.load(h_path)

    J = np.load(os.path.join(ROOT, "J.npy"))
    qaoa = QAOA(J)
    X, _ = build_features(J, h)

    ckpt = torch.load(os.path.join(DATA, "model.pt"), map_location="cpu",
                      weights_only=True)
    net = QAOAAngleNet(in_dim=ckpt["in_dim"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    xt = torch.tensor(X, dtype=torch.float32)
    ht = torch.tensor(h, dtype=torch.float32)

    with torch.no_grad():
        g0, b0 = net.angles(xt)
        p_before = qaoa.p_ground(ht, g0, b0).mean().item()
    print(f"P(ground) чистой сети (без полировки): {p_before:.4f}")

    # полировка: поинстансный Adam от предсказания сети
    g = g0.detach().clone().requires_grad_(True)
    b = b0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([g, b], lr=POLISH_LR)
    t0 = time.time()
    for step in range(args.polish):
        opt.zero_grad()
        loss = -qaoa.p_ground(ht, g, b).mean()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"  polish {step}/{args.polish}: mean P(ground) = {-loss.item():.4f}",
                  flush=True)
    dt = time.time() - t0

    with torch.no_grad():
        p_after = qaoa.p_ground(ht, g, b).mean().item()
    print(f"P(ground) после полировки ({args.polish} шагов): {p_after:.4f} "
          f"({dt:.1f} c)")

    g_np, b_np = g.detach().numpy(), b.detach().numpy()
    assert len(np.unique(g_np, axis=0)) > 1, "углы должны зависеть от h"
    cols = (["id"] + [f"gamma_{k}" for k in range(5)]
            + [f"beta_{k}" for k in range(5)])
    out = np.concatenate(
        [np.arange(len(h))[:, None], g_np, b_np], axis=1)
    np.savetxt(args.out, out, delimiter=",",
               header=",".join(cols), comments="", fmt=["%d"] + ["%r"] * 10)
    print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
