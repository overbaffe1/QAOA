"""Обучение сети end-to-end через дифференцируемый симулятор QAOA.

Главная потеря — -P(ground)(h, f(h)): она не знает о «неуникальности
оптимальных углов» (сеть может выбрать любой из эквивалентных по качеству
бассейнов), в отличие от L2-регрессии на метки. Вспомогательные потери
(энергия и конфигурация основного состояния) дают физический сигнал и
улучшают обобщение.

Данные:
  * 450 реальных h (50 выделены в holdout-валидацию);
  * 1000 синтетических h ~ U[-1,1]^12 (распределение h_test неизвестно
    точнее, чем у h_train — синтетика расширяет покрытие пространства h);
  * валидация: 50 реальных + 500 синтетических.

Инициализация голов углов — средними значениями «меток» (результат полной
поинстансной оптимизации), чтобы сеть стартовала в окрестности хороших углов.

Запуск:  python train.py
Артефакт: data/model.pt (лучшая по валидации чекпоинт) + data/history.csv
"""
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (AUX_W, BATCH, DATA, LR, N_SYNTH_TRAIN, N_SYNTH_VAL,  # noqa: E402
                    N_VAL_REAL, ROOT, SEED, STEPS)
from features import build_features  # noqa: E402
from model import QAOAAngleNet  # noqa: E402
from QAOA import QAOA  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def polish(qaoa, h, g0, b0, steps=30, lr=0.05):
    """Короткая поинстансная доводка углов от начального приближения."""
    g = g0.detach().clone().requires_grad_(True)
    b = b0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([g, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = -qaoa.p_ground(h, g, b).mean()
        loss.backward()
        opt.step()
    return g.detach(), b.detach(), -loss.item()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    J = np.load(os.path.join(ROOT, "J.npy"))
    h_all = np.load(os.path.join(ROOT, "h_train.npy"))
    qaoa = QAOA(J, device=DEVICE)
    print(f"device: {DEVICE}")

    # ---------- разбиение и синтетика ----------
    idx = np.arange(len(h_all))
    np.random.default_rng(SEED).shuffle(idx)
    val_idx, tr_idx = idx[:N_VAL_REAL], idx[N_VAL_REAL:]

    rng = np.random.default_rng(SEED + 1)
    h_syn_tr = rng.uniform(-1, 1, size=(N_SYNTH_TRAIN, 12))
    h_syn_val = rng.uniform(-1, 1, size=(N_SYNTH_VAL, 12))

    h_tr = np.concatenate([h_all[tr_idx], h_syn_tr], axis=0)
    h_val = np.concatenate([h_all[val_idx], h_syn_val], axis=0)

    X_tr, gs_tr = build_features(J, h_tr)
    X_val, gs_val = build_features(J, h_val)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    htr = torch.tensor(h_tr, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    hval = torch.tensor(h_val, dtype=torch.float32, device=DEVICE)
    emin_tr = torch.tensor(gs_tr["Emin"].reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    sstar_tr = torch.tensor((gs_tr["s_star"] + 1) / 2, dtype=torch.float32, device=DEVICE)
    emin_val = torch.tensor(gs_val["Emin"].reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    sstar_val = torch.tensor((gs_val["s_star"] + 1) / 2, dtype=torch.float32, device=DEVICE)

    # ---------- сеть (инициализация голов углов по меткам) ----------
    net = QAOAAngleNet(in_dim=Xt.shape[1]).to(DEVICE)
    labels_path = os.path.join(DATA, "labels.npz")
    if os.path.exists(labels_path):
        lab = np.load(labels_path)
        for k in range(5):
            net.g_heads[k].bias.data.fill_(float(lab["gamma"][:, k].mean()))
            net.b_heads[k].bias.data.fill_(float(lab["beta"][:, k].mean()))
        print("головы углов инициализированы средними значениями меток")
    else:
        print("ВНИМАНИЕ: data/labels.npz не найдено — heads без инициализации по меткам")

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ckpt = os.path.join(DATA, "model.pt")
    hist = os.path.join(DATA, "history.csv")
    best_val = -1.0
    B = len(h_tr)
    rng_t = np.random.default_rng(SEED + 2)

    def val_p_ground():
        with torch.no_grad():
            g, b, _, _ = net(Xv)
            return qaoa.p_ground(hval, g, b).mean().item()

    with open(hist, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "val_p_ground", "val_p_ground_polished", "loss"])
        for step in range(STEPS):
            sel = rng_t.choice(B, size=BATCH, replace=False)
            x, hh = Xt[sel], htr[sel]
            g, b, e, s = net(x)
            p = qaoa.p_ground(hh, g, b)
            aux = torch.nn.functional.mse_loss(e, emin_tr[sel]) + \
                torch.nn.functional.binary_cross_entropy(
                    s.clamp(1e-6, 1 - 1e-6), sstar_tr[sel])
            loss = -p.mean() + AUX_W * aux
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 1000 == 0 or step == STEPS - 1:
                v = val_p_ground()
                gv, bv, _, _, = net(Xv)
                _, _, v_pol = polish(qaoa, hval[:N_VAL_REAL],
                                     gv[:N_VAL_REAL], bv[:N_VAL_REAL], steps=30)
                w.writerow([step, f"{v:.5f}", f"{v_pol:.5f}", f"{loss.item():.5f}"])
                print(f"step {step:5d}: loss={loss.item():.4f}  "
                      f"val P(ground)={v:.4f}  (с полировкой, 50 real)={v_pol:.4f}",
                      flush=True)
                if v > best_val:
                    best_val = v
                    torch.save({"state_dict": net.state_dict(),
                                "in_dim": Xt.shape[1]}, ckpt)
    print(f"\nлучший val P(ground) = {best_val:.4f}, сохранено: {ckpt}")


if __name__ == "__main__":
    main()
