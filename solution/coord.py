"""COORD — поочерёдный поиск: по одному углу за раз.

Точно то, что надо: фиксируем остальные 9 углов, перебираем первый
по ВСЕМУ кругу 2π (--grid точек), все 500 инстансов одним батчем,
берём лучшее per-instance (ratchet). Затем так же 2-й угол, ...,
10-й. Один круг = «раунд». Повторяем до --passes или пока среднее
не перестанет расти.

Плюсы против Adam/L-BFGS:
  - видит весь круг 2π — не «застревает рядом с текущим значением»;
  - без learning rate — нечего тюнить;
  - плотно: --grid 48 точек/угло — узкие пики не проваливаются
    (шаг Adam lr=0.05 их перескакивает).
Слабость (честно): двигается только «вдоль осей» — диагональные
долины переезжает медленно (лечится раундами) и в локальный холм
застрянет как любой метод → идеальная пара: lucky.py (kicks = удача)
затем coord.py (дожим до вершины). ~2-4 мин на все 500 (GPU),
ratchet: выход никогда хуже входного CSV.

Запуск:
  python coord.py --h h_train.npy --from lucky.csv --out lucky2.csv
  # точнее: --grid 64 --passes 4
  # и финиш параллельным L-BFGS: --lbfgs 30 --procs 4
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QAOA import QAOA, P  # noqa: E402
from brain import N_ANGLES  # noqa: E402
from megabrute import split10  # noqa: E402
from lucky import save_csv, lbfgs_parallel, p_eval  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def coord_pass(qaoa, ht, a10, p, grid, tag):
    """Один круг: по каждому углу перебор всего 2π, ratchet внутри."""
    B = a10.shape[0]
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))
    a_best, p_best = a10.copy(), p.copy()
    gv = np.linspace(0.0, 2.0 * np.pi, grid, endpoint=False)
    for k in range(N_ANGLES):
        for c0 in range(0, grid, 8):
            G = gv[c0:c0 + 8]
            a10c = np.tile(a_best, (len(G), 1, 1))
            a10c[:, :, k] = G[:, None]
            a10c = a10c.reshape(-1, N_ANGLES)
            ht_c = torch.cat([ht] * len(G), dim=0)
            g, b = split10(a10c)
            with torch.no_grad():
                pc = qaoa.p_ground(ht_c, g, b).cpu().numpy()
            pc = pc.reshape(len(G), B)
            am = pc.argmax(axis=0)
            pm = pc.max(axis=0)
            upd = pm > p_best
            a_best[upd] = a10c.reshape(len(G), B, N_ANGLES)[am[upd],
                                                      np.where(upd)[0]]
            p_best[upd] = pm[upd]
        print(f"  {tag} угол {k + 1}/{N_ANGLES}: mean = {p_best.mean():.4f}",
              flush=True)
    return a_best, p_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="h_train.npy")
    ap.add_argument("--from", dest="src", default="lucky.csv")
    ap.add_argument("--out", default="coord.csv")
    ap.add_argument("--grid", type=int, default=48,
                    help="точек по всему кругу 2π на угол")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--min-improve", type=float, default=5e-5)
    ap.add_argument("--lbfgs", type=int, default=0,
                    help="финальный L-BFGS (0 = без)")
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h_path = args.h if os.path.exists(args.h) else os.path.join(root, args.h)
    src_path = (args.src if os.path.exists(args.src)
                else os.path.join(root, args.src))
    h = np.load(h_path)
    j_path = os.path.join(root, "J.npy")
    qaoa = QAOA(np.load(j_path), device=DEVICE)
    ht = torch.tensor(h, dtype=torch.float32, device=DEVICE)

    data = np.genfromtxt(src_path, delimiter=",", skip_header=1)
    ids, a = data[:, 0].astype(int), data[:, 1:11].astype(np.float32)
    assert a.shape[0] == h.shape[0], f"CSV {a.shape[0]} != h {h.shape[0]}"
    p = p_eval(qaoa, ht, a)
    p0 = p.mean()
    print(f"старт: mean P(ground) = {p0:.4f} (из {os.path.basename(args.src)})",
          flush=True)
    print(f"поиск: {args.grid} точек/угол x 10 углов x {args.passes} раундов",
          flush=True)

    for r in range(args.passes):
        t0 = time.time()
        p_prev = p.mean()
        a, p = coord_pass(qaoa, ht, a, p, args.grid, f"раунд {r + 1}")
        impr = p.mean() - p_prev
        print(f"  раунд {r + 1}: mean = {p.mean():.4f} "
              f"(+{impr:.4f} за раунд, +{p.mean() - p0:.4f} от старта, "
              f"{time.time() - t0:.0f} c)", flush=True)
        if r > 0 and impr <= args.min_improve:
            print(f"  сходимость (рост раунда < {args.min_improve}) — стоп",
                  flush=True)
            break

    if args.lbfgs > 0:
        res = lbfgs_parallel(h, a, np.arange(a.shape[0]), args.lbfgs,
                             j_path, args.procs, "L-BFGS финал")
        n_imp = 0
        for i in range(a.shape[0]):
            out, pn = res[i]
            if pn > p[i] + 1e-12:
                a[i], p[i] = out, pn
                n_imp += 1
        print(f"  L-BFGS: улучшено {n_imp}/{a.shape[0]} | mean = "
              f"{p.mean():.4f}", flush=True)

    save_csv(args.out, a, ids)
    tail = np.argsort(p)[:10]
    print(f"\nИТОГ: mean P(ground) = {p.mean():.4f}  (старт {p0:.4f})",
          flush=True)
    print("  tail: " + " ".join(f"#{i}:{p[i]:.3f}" for i in tail),
          flush=True)
    print(f"гарантия: {args.out} никогда хуже {args.src} (ratchet).",
          flush=True)
    print(f"файл: {args.out}", flush=True)


if __name__ == "__main__":
    main()
