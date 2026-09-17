"""Дистилляция: докармливаем «мозг» обратно в сеть.

Замкнутый цикл улучшения через мозг (концепция — Iteration-Free QAOA
using neural networks, 2024: «чем больше прошлых оптимизаций, тем меньше
итераций нужно сети»):

  1) Тяжёлый поиск:  infer.py --profile brain  ->  лучшие углы по инстансам
     копятся в brain.pkl;
  2) distill.py — дообучает сеть на схемах мозга:
        loss = -P(ground)(h, f(h))  +  W_ANG * <угловое расстояние до углов
                                         мозга (с учётом периодичности)>;
  3) Следующий раунд поиска стартует от УМЕЮЩЕЙ сети -> вся кривая
     поднимается. Мозг при этом остаётся — история не теряется.

Запуск:  python distill.py
Артефакты: data/model.pt (обновлена), data/model_backup_pre_distill.pt
(резервная копия ДО дистилляции, создаётся один раз).
"""
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import QAOABrain  # noqa: E402
from config import DATA, ROOT, SEED  # noqa: E402
from features import build_features  # noqa: E402
from model import QAOAAngleNet  # noqa: E402
from QAOA import QAOA  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, LR, BATCH, W_ANG = 2000, 1e-4, 64, 0.05


def ang_dist(a, b):
    """Гладкая «кордовая» дистанция на круге: 0, если углы различаются
    на 2*pi*k, максимум 2, производная ограничена везде (в отличие от
    arccos(cos), чей градиент расходится при совпадении углов)."""
    return 2.0 * torch.sin(0.5 * (a - b)) ** 2


def main():
    torch.manual_seed(SEED)
    brain_path = os.path.join(DATA, "brain.pkl")
    brain = QAOABrain.load(brain_path)
    if not brain.patterns:
        sys.exit("brain.pkl пуст — сначала: python infer.py --profile brain")

    h_all = np.load(os.path.join(ROOT, "h_train.npy"))
    targets = [(i, brain.best_for(h)) for i, h in enumerate(h_all)]
    targets = [(i, a) for i, a in targets if a is not None]
    if not targets:
        sys.exit("в мозге нет схем для h_train")
    idx = np.array([t[0] for t in targets])
    ang = np.array([t[1] for t in targets])
    print(f"дистилляция: {len(idx)} инстансов с целевыми углами из мозга")

    J = np.load(os.path.join(ROOT, "J.npy"))
    qaoa = QAOA(J, device=DEVICE)
    h = h_all[idx]
    X, _ = build_features(J, h)

    ckpt_path = os.path.join(DATA, "model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net = QAOAAngleNet(in_dim=ckpt["in_dim"]).to(DEVICE)
    net.load_state_dict(ckpt["state_dict"])

    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    ht = torch.tensor(h, dtype=torch.float32, device=DEVICE)
    At = torch.tensor(ang, dtype=torch.float32, device=DEVICE)

    def report(tag):
        with torch.no_grad():
            g, b, _, _ = net(Xt)
            p = qaoa.p_ground(ht, g, b).mean().item()
        print(f"{tag}: P(ground) чистой сети на {len(idx)} инстансах "
              f"мозга = {p:.4f}", flush=True)

    report("до дистилляции")

    backup = os.path.join(DATA, "model_backup_pre_distill.pt")
    if not os.path.exists(backup):
        shutil.copy(ckpt_path, backup)
        print(f"резервная копия модели: {backup}")

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    B = len(h)
    rng = np.random.default_rng(SEED + 7)
    t0 = time.time()
    for step in range(STEPS):
        sel = rng.choice(B, size=min(BATCH, B), replace=False)
        g, b, e, s = net(Xt[sel])
        p = qaoa.p_ground(ht[sel], g, b)
        dist = ang_dist(torch.cat([g, b], 1), At[sel]).mean()
        loss = -p.mean() + W_ANG * dist
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == STEPS - 1:
            print(f"step {step}: loss={loss.item():.4f} "
                  f"ang_dist={dist.item():.4f}  ({time.time() - t0:.0f} c)",
                  flush=True)
    torch.save({"state_dict": net.state_dict(), "in_dim": ckpt["in_dim"]},
               ckpt_path)
    report("после дистилляции")
    print(f"сохранено: {ckpt_path}")


if __name__ == "__main__":
    main()
