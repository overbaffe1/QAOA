"""Доводка ГОТОВОЙ посылки: per-instance L-BFGS (+ опционально Adam) поверх
углов из CSV/npz. Ничего не портит: для каждого инстанса угол принимается
только если P(ground) стал не хуже («keep best»).

Зачем:
  * лидерборд (h_train) можно растить без лимита — довёл готовый CSV, получил
    прибавку, загрузил;
  * в финале (h_test, лимит 600 с) это страховка: `--time-budget 500` —
    доводка прервётся на том инстансе, где бюджет исчерпан, и всё равно
    запишет валидный CSV;
  * работает на CPU (память — один инстанс за раз), поэтому доводить можно
    даже без GPU, просто медленнее.

Примеры:
  # довести метки/посылку на h_train (без ограничения времени)
  python polish_csv.py --csv ../submission.csv --h ../h_train.npy \
      --out ../submission_polished.csv --refine-iters 50

  # из data/labels.npz, с бюджетом 500 с
  python polish_csv.py --npz ../data/labels.npz --h ../h_train.npy \
      --out ../submission_polished.csv --refine-iters 50 --time-budget 500
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from config import ROOT as CFG_ROOT  # noqa: E402
from QAOA import QAOA, P  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BUDGET_RESERVE = 15.0      # запас на оценку и запись CSV
EVAL_CHUNK = 128           # инстансов на один батч оценки (экономия памяти)


def load_angles(csv=None, npz=None):
    """(gamma, beta) из submission.csv или из npz (labels.npz)."""
    if csv:
        with open(csv, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            rows = [[float(c) for c in ln.strip().split(",")]
                    for ln in f if ln.strip()]
        a = np.asarray(rows, dtype=np.float64)
        if header[0].strip().lower() == "id":
            a = a[:, 1:]
        if a.shape[1] != 2 * P:
            sys.exit(f"{csv}: ожидалось {2 * P} колонок углов, найдено {a.shape[1]}")
        return a[:, :P], a[:, P:]
    a = np.load(npz)
    return np.asarray(a["gamma"], dtype=np.float64), \
        np.asarray(a["beta"], dtype=np.float64)


def p_all(qaoa, ht, g, b):
    """P(ground) по всем инстансам (кусками, чтобы не съесть память)."""
    out = np.empty(len(ht), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(ht), EVAL_CHUNK):
            sl = slice(i, i + EVAL_CHUNK)
            out[sl] = qaoa.p_ground(ht[sl], g[sl], b[sl]).cpu().numpy()
    return out


def write_csv(path, g, b):
    cols = (["id"] + [f"gamma_{k}" for k in range(P)]
            + [f"beta_{k}" for k in range(P)])
    out = np.concatenate([np.arange(len(g))[:, None],
                          np.asarray(g, dtype=np.float64),
                          np.asarray(b, dtype=np.float64)], axis=1)
    np.savetxt(path, out, delimiter=",", header=",".join(cols),
               comments="", fmt=["%d"] + ["%.12f"] * (2 * P))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="submission.csv (id, gamma_0..4, beta_0..4)")
    src.add_argument("--npz", help="npz с ключами gamma, beta (например labels.npz)")
    ap.add_argument("--h", default="h_train.npy")
    ap.add_argument("--out", default="submission_polished.csv")
    ap.add_argument("--refine-iters", type=int, default=50,
                    help="итераций L-BFGS на инстанс (0 = выключить)")
    ap.add_argument("--fine", type=int, default=0,
                    help="шагов Adam (lr 0.01) батчем перед L-BFGS")
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="жёсткий бюджет секунд (0 = без ограничения)")
    args = ap.parse_args()

    t_start = time.time()
    deadline = (t_start + args.time_budget) if args.time_budget > 0 else None

    h_path = args.h if os.path.exists(args.h) else os.path.join(CFG_ROOT, args.h)
    if not os.path.exists(h_path):
        sys.exit(f"файл {h_path} не найден")
    h = np.load(h_path)
    g_np, b_np = load_angles(csv=args.csv, npz=args.npz)
    if len(g_np) != len(h):
        sys.exit(f"строк углов {len(g_np)} != инстансов {len(h)}")
    print(f"вход: {args.csv or args.npz}, N = {len(h)}, device = {DEVICE}",
          flush=True)

    J = np.load(os.path.join(CFG_ROOT, "J.npy"))
    qaoa = QAOA(J, device=DEVICE)
    ht = torch.tensor(h, dtype=torch.float32, device=DEVICE)

    g = torch.tensor(g_np, dtype=torch.float32, device=DEVICE)
    b = torch.tensor(b_np, dtype=torch.float32, device=DEVICE)
    p0 = p_all(qaoa, ht, g, b)
    print(f"до доводки: mean = {p0.mean():.6f}, median = {np.median(p0):.6f}",
          flush=True)

    # ---------- батчевая Adam-доводка мелким lr (необязательно) ----------
    if args.fine > 0:
        gg = g.clone().requires_grad_(True)
        bb = b.clone().requires_grad_(True)
        opt = torch.optim.Adam([gg, bb], lr=0.01)
        for s in range(args.fine):
            opt.zero_grad()
            loss = -qaoa.p_ground(ht, gg, bb).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p_fine = p_all(qaoa, ht, gg, bb)
        upd = p_fine > p0
        g = torch.where(torch.tensor(upd, device=DEVICE).unsqueeze(1),
                        gg.detach(), g)
        b = torch.where(torch.tensor(upd, device=DEVICE).unsqueeze(1),
                        bb.detach(), b)
        p0 = np.maximum(p0, p_fine)
        print(f"после Adam fine ({args.fine} шагов): mean = {p0.mean():.6f} "
              f"(улучшилось {int(upd.sum())}/{len(h)})", flush=True)

    # ---------- per-instance L-BFGS («final refinement») ----------
    g_out = g.detach().cpu().numpy().astype(np.float64)
    b_out = b.detach().cpu().numpy().astype(np.float64)
    p_out = p0.copy()
    n_ref = n_better = 0
    if args.refine_iters > 0:
        t0 = time.time()
        for i in range(len(h)):
            if deadline is not None and i > 0:
                left = deadline - time.time()
                per = (time.time() - t0) / i
                if left < per + BUDGET_RESERVE:
                    print(f"[бюджет] доводка остановлена на инстансе {i}/{len(h)} "
                          f"(~{per * (len(h) - i):.0f} c не хватило)", flush=True)
                    break
            gi = torch.tensor(g_out[i], dtype=torch.float32, device=DEVICE,
                              requires_grad=True)
            bi = torch.tensor(b_out[i], dtype=torch.float32, device=DEVICE,
                              requires_grad=True)

            def closure():
                opt.zero_grad()
                loss = -qaoa.p_ground(ht[i:i + 1], gi.unsqueeze(0),
                                      bi.unsqueeze(0)).mean()
                loss.backward()
                return loss

            opt = torch.optim.LBFGS([gi, bi], lr=1.0,
                                    max_iter=args.refine_iters, history_size=10)
            opt.step(closure)
            with torch.no_grad():
                p_new = qaoa.p_ground(ht[i:i + 1], gi.unsqueeze(0),
                                      bi.unsqueeze(0)).item()
            n_ref += 1
            if p_new > p_out[i]:                 # keep best — хуже не делаем
                g_out[i] = gi.detach().cpu().numpy()
                b_out[i] = bi.detach().cpu().numpy()
                p_out[i] = p_new
                n_better += 1
            if (i + 1) % 50 == 0:
                print(f"  доведено {i + 1}/{len(h)}: mean = {p_out.mean():.6f} "
                      f"({time.time() - t0:.0f} c)", flush=True)
        print(f"L-BFGS: {n_ref} инстансов за {time.time() - t0:.0f} c, "
              f"улучшилось {n_better}", flush=True)

    write_csv(args.out, g_out, b_out)
    p_chk = p_all(qaoa, ht, torch.tensor(g_out, dtype=torch.float32, device=DEVICE),
                  torch.tensor(b_out, dtype=torch.float32, device=DEVICE))
    print(f"\nпосле доводки: mean = {p_chk.mean():.6f}, "
          f"median = {np.median(p_chk):.6f}, min = {p_chk.min():.6f}, "
          f"max = {p_chk.max():.6f}")
    print(f"сырое среднее (лидерборд) = "
          f"{np.asarray(p_chk, dtype=np.float64).mean():.16f}")
    print(f"сохранено: {args.out}")
    print(f"время всего: {time.time() - t_start:.1f} c"
          + (f" / бюджет {args.time_budget:.0f} c" if deadline else ""))
    if p_chk.mean() < p0.mean() - 1e-9:
        print("ВНИМАНИЕ: результат хуже входного — оставляйте входной CSV")


if __name__ == "__main__":
    main()
