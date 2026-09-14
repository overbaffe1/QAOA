"""Инференс: модель -> углы -> СИЛЬНАЯ поинстансная полировка -> submission.csv.

Этапы:
  1. Сеть f(h) -> (gamma, beta)  (один проход на все 500 инстансов);
  2. Сильная полировка (поинстансно, батч = все инстансы):
       * рестарт 0 — предсказание сети,
       * рестарт 1 — предсказание сети + пертурбация N(0, 0.3),
       * рестарты 2..R-1 — случайные углы в полном диапазоне
         gamma ~ U[0, 2pi], beta ~ U[0, pi];
     каждый рестарт — POLISH_STEPS шагов Adam, затем финальная доводка
     мелким lr от лучшего; для каждого инстанса оставляем лучший набор
     углов по P(ground). Тысячи итераций классического цикла QAOA
     заменяются десятками — при потере лишь долей процента P(ground)
     против полной оптимизации.
  3. Запись submission.csv (id, gamma_0..gamma_4, beta_0..beta_4).

Время (500 инстансов): на GPU (Colab) ~2-5 мин; на CPU — часы.
Лимит задачи: 10 минут (инференс на GPU с запасом укладывается).

Профили:
  full — для финальной посылки (GPU): 8 рестартов x 300 шагов + доводка;
  fast — быстрая самопроверка на CPU: 3 рестарта x 100 шагов.

Запуск:
  python infer.py --h h_test.npy --out submission.csv              # full
  python infer.py --h h_train.npy --out sub.csv --profile fast     # самопроверка
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (DATA, POLISH_FINE_LR, POLISH_FINE_STEPS,  # noqa: E402
                    POLISH_FINE_STEPS_FAST, POLISH_LR, POLISH_RESTARTS,
                    POLISH_RESTARTS_FAST, POLISH_STEPS, POLISH_STEPS_FAST,
                    ROOT)
from features import build_features  # noqa: E402
from model import QAOAAngleNet  # noqa: E402
from QAOA import QAOA, P  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_restart(qaoa, ht, g, b, steps, lr, fine_steps, fine_lr, tag):
    """Adam-оптимизация (g, b) с финальной доводкой мелким lr. Возвращает
    (g, b, p_ground по инстансам)."""
    B = ht.shape[0]
    g = g.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True)
    t0 = time.time()
    opt = torch.optim.Adam([g, b], lr=lr)
    for s in range(steps):
        opt.zero_grad()
        loss = -qaoa.p_ground(ht, g, b).mean()
        loss.backward()
        opt.step()
        if steps >= 100 and s % 100 == 0:
            print(f"  {tag} {s}/{steps}: mean P(ground) = {-loss.item():.4f}",
                  flush=True)
    if fine_steps:
        opt = torch.optim.Adam([g, b], lr=fine_lr)
        for s in range(fine_steps):
            opt.zero_grad()
            loss = -qaoa.p_ground(ht, g, b).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        p = qaoa.p_ground(ht, g, b)
    print(f"{tag}: mean P(ground) = {p.mean().item():.4f} "
          f"({time.time() - t0:.0f} c)", flush=True)
    return g.detach(), b.detach(), p


def polish_strong(qaoa, ht, g0, b0, restarts, steps, fine_steps,
                  fine_lr, lr, seed=42):
    """Много-рестарт поинстансная полировка, best-of по P(ground)."""
    B = ht.shape[0]
    torch.manual_seed(seed)
    best_g = g0.detach().clone()
    best_b = b0.detach().clone()
    best_p = -torch.ones(B, device=ht.device)

    for r in range(restarts):
        if r == 0:
            g, b = g0, b0
            tag = f"рестарт {r + 1}/{restarts} (сеть)"
        elif r == 1:
            g = g0 + 0.3 * torch.randn_like(g0)
            b = b0 + 0.3 * torch.randn_like(b0)
            tag = f"рестарт {r + 1}/{restarts} (сеть + шум)"
        else:
            g = torch.rand(B, P, device=ht.device) * 2 * np.pi
            b = torch.rand(B, P, device=ht.device) * np.pi
            tag = f"рестарт {r + 1}/{restarts} (случайный)"
        gs, bs, p = run_restart(qaoa, ht, g, b, steps, lr,
                                fine_steps, fine_lr, tag)
        upd = (p > best_p).unsqueeze(1)
        best_g = torch.where(upd, gs, best_g)
        best_b = torch.where(upd, bs, best_b)
        best_p = torch.maximum(p, best_p)
        print(f"  -> best-of mean P(ground) = {best_p.mean().item():.4f}",
              flush=True)
    return best_g, best_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="h_test.npy", help="файл с h")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--profile", choices=["full", "fast"], default="full",
                    help="full — финальная (GPU), fast — самопроверка (CPU)")
    ap.add_argument("--restarts", type=int, default=None,
                    help="переопределить число рестартов (по умолчанию — из профиля)")
    ap.add_argument("--steps", type=int, default=None,
                    help="переопределить число шагов Adam на рестарт")
    ap.add_argument("--fine", type=int, default=None,
                    help="переопределить число шагов финальной доводки")
    ap.add_argument("--limit", type=int, default=0,
                    help="обработать только первые N (только для теста)")
    args = ap.parse_args()

    if args.profile == "full":
        R, S, FS = POLISH_RESTARTS, POLISH_STEPS, POLISH_FINE_STEPS
    else:
        R, S, FS = POLISH_RESTARTS_FAST, POLISH_STEPS_FAST, POLISH_FINE_STEPS_FAST
    R = args.restarts if args.restarts is not None else R
    S = args.steps if args.steps is not None else S
    FS = args.fine if args.fine is not None else FS
    print(f"profile={args.profile}: restarts={R}, steps={S}, fine={FS}")

    h_path = args.h if os.path.exists(args.h) else os.path.join(ROOT, args.h)
    if not os.path.exists(h_path):
        sys.exit(f"файл {h_path} не найден")
    h = np.load(h_path)
    if args.limit > 0:
        h = h[:args.limit]
        print(f"(ТЕСТОВЫЙ РЕЖИМ: только первые {len(h)} из {args.limit}+)")

    J = np.load(os.path.join(ROOT, "J.npy"))
    qaoa = QAOA(J, device=DEVICE)
    X, _ = build_features(J, h)

    ckpt = torch.load(os.path.join(DATA, "model.pt"), map_location="cpu",
                      weights_only=True)
    net = QAOAAngleNet(in_dim=ckpt["in_dim"]).to(DEVICE)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    ht = torch.tensor(h, dtype=torch.float32, device=DEVICE)
    print(f"device: {DEVICE}, N = {len(h)}")

    with torch.no_grad():
        g0, b0 = net.angles(xt)
        p_before = qaoa.p_ground(ht, g0, b0).mean().item()
    print(f"P(ground) чистой сети (без полировки): {p_before:.4f}")

    t_total = time.time()
    g, b = polish_strong(qaoa, ht, g0, b0, R, S, FS, POLISH_FINE_LR, POLISH_LR)
    dt = time.time() - t_total

    with torch.no_grad():
        p_after = qaoa.p_ground(ht, g, b).mean().item()
    print(f"\nP(ground) после полировки: {p_after:.4f}  (сеть давала {p_before:.4f})")
    print(f"время полировки: {dt:.1f} c (лимит 600 c)")

    g_np, b_np = g.cpu().numpy(), b.cpu().numpy()
    assert len(np.unique(g_np, axis=0)) > 1, "углы должны зависеть от h"
    cols = (["id"] + [f"gamma_{k}" for k in range(5)]
            + [f"beta_{k}" for k in range(5)])
    out = np.concatenate(
        [np.arange(len(h))[:, None], g_np, b_np], axis=1)
    np.savetxt(args.out, out, delimiter=",",
               header=",".join(cols), comments="", fmt=["%d"] + ["%.12f"] * 10)
    print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
