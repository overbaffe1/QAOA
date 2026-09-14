"""Генерация «меток»: поинстансная оптимизация углов QAOA для h_train.

Каждому из 500 векторов h — свои (gamma, beta); Adam, несколько независимых
случайных инициализаций (рестартов), для каждого инстанса оставляем лучший
рестарт по P(ground). Батч = все 500 инстансов одновременно (симулятор
поддерживает поинстансные углы), поэтому всё векторизуется.

Артефакт: data/labels.npz — gamma (500,5), beta (500,5), p_ground (500).
p_ground здесь — «потолок» (качество полной поинстансной оптимизации) для
каждого инстанса.

Запуск:  python generate_labels.py [--steps 250] [--restarts 3] [--chunk 64]
Время: ~15-20 мин на CPU (зависит от машины), секунды-минуты на GPU.

--chunk N — считать не все 500 инстансов одним батчем, а кусками по N.
Углы у каждого инстанса свои, инстансы между собой не связаны, поэтому
результат практически тот же (Adam нормирует градиент на размер батча;
замер на 8 инстансах и 100 шагах: среднее P(ground) отличается меньше чем
на 1%, углы — меньше чем на 0.03 рад, корреляция по инстансам 0.9998),
а память падает пропорционально N. Нужно на машинах с малой RAM: батч 500 x 12 кубитов
x p=5 на backward требует больше 3 ГБ и убивается OOM-киллером; --chunk 64
держится в ~0.5 ГБ.
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run(steps=250, restarts=3, lr=LABEL_LR, seed=42, device=DEVICE, chunk=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    J = np.load(os.path.join(ROOT, "J.npy"))
    h = np.load(os.path.join(ROOT, "h_train.npy"))
    qaoa = QAOA(J, device=device)

    B = len(h)
    ckpt = os.path.join(DATA, "labels.npz")

    best_g = np.zeros((B, P))
    best_b = np.zeros((B, P))
    best_p = np.full(B, -1.0)
    done = np.zeros(B, dtype=bool)

    if chunk and chunk > 0:
        blocks = [np.arange(i, min(i + chunk, B)) for i in range(0, B, chunk)]
    else:
        blocks = [np.arange(B)]
    print(f"B = {B}, рестартов = {restarts}, шагов = {steps}, "
          f"кусков = {len(blocks)} (по {len(blocks[0])} инстансов)", flush=True)

    def save():
        np.savez(ckpt, gamma=best_g, beta=best_b, p_ground=best_p)

    for r in range(restarts):
        # инициализация сразу на все инстансы (как раньше), оптимизация — кусками
        g_all = torch.rand(B, P) * 0.9 + 0.05
        b_all = torch.rand(B, P) * 0.9 + 0.05
        t0 = time.time()
        for bi, idx in enumerate(blocks):
            ht = torch.tensor(h[idx], dtype=torch.float32, device=device)
            g = g_all[idx].clone().to(device).requires_grad_(True)
            b = b_all[idx].clone().to(device).requires_grad_(True)
            opt = torch.optim.Adam([g, b], lr=lr)
            for step in range(steps):
                opt.zero_grad()
                loss = -qaoa.p_ground(ht, g, b).mean()
                loss.backward()
                opt.step()
                if step % 100 == 0 and bi == 0:
                    print(f"  restart {r + 1}/{restarts} кусок {bi + 1}"
                          f"/{len(blocks)} step {step}: "
                          f"mean P(ground) = {-loss.item():.4f}", flush=True)
            with torch.no_grad():
                p = qaoa.p_ground(ht, g, b).cpu().numpy()
            gn = g.detach().cpu().numpy()
            bn = b.detach().cpu().numpy()
            upd = p > best_p[idx]
            best_g[idx] = np.where(upd[:, None], gn, best_g[idx])
            best_b[idx] = np.where(upd[:, None], bn, best_b[idx])
            best_p[idx] = np.where(upd, p, best_p[idx])
            done[idx] = True
            del ht, g, b, opt
            save()                      # чекпоинт после каждого куска
            print(f"  restart {r + 1}/{restarts}: кусок {bi + 1}/{len(blocks)} "
                  f"готов ({time.time() - t0:.0f} c от начала рестарта), "
                  f"mean по готовым = {best_p[done].mean():.4f}", flush=True)

        dt = time.time() - t0
        print(f"restart {r + 1}/{restarts}: mean = {best_p.mean():.4f}, "
              f"{dt:.0f} c", flush=True)
        save()

    print(f"\nИТОГО по {B} инстансам: mean = {best_p.mean():.4f}, "
          f"median = {np.median(best_p):.4f}, "
          f"min = {best_p.min():.4f}, max = {best_p.max():.4f}", flush=True)
    print(f"сохранено: {ckpt}", flush=True)
    save()
    return best_g, best_b, best_p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=0,
                    help="размер куска по инстансам (0 = все одним батчем; "
                         "64 — для машин с малой RAM)")
    ap.add_argument("--lr", type=float, default=LABEL_LR)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(steps=args.steps, restarts=args.restarts, lr=args.lr, seed=args.seed,
        chunk=args.chunk)
