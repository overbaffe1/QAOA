"""DEEP GRIND — долгий поинстансный «глубокий» прогон h_train.

Почему это путь к 0.4-0.9:
  Лидерборд — RAW P(ground) на h_train БЕЗ лимита времени. Топы
  (0.5-0.95) = поинстансный глобальный поиск с ОГРОМНЫМ бюджетом
  function-evaluations на инстанс (1e4-1e6 FE), тогда как «широкий»
  поиск megabrute даёт ~1e3 FE на инстанс за раунд. Потолок P(ground)
  с инстанса (p=5, 12 кубитов) часто 0.6-0.95 — при глубоком поиске.

Каждый раунд = воронка coarse-to-fine («быстро проверить 100-1000
сидов, отобрать, потом дожимать»):
  1) на инстанс --seeds сидов: случайные + KICKS текущего лучшего
     (σ=0.3..3.0) + kNN-трансфер+шум + топ-схемы+шум. Инстансы с
     P < --focus получают ещё ПОЛНЫЙ набор сидов (фокус на хвост —
     именно он держит среднее).
  2) COARSE: все сиды x все инстансы чанками, Adam --screen-steps
     (48 сидов ~3 мин GPU; 1000 сидов ~1 ч) -> P каждого сида.
  3) per-instance top-k выживших (по P скрининга).
  4) FINE: выжившие -> Adam --fine-steps + --fine (чанками).
  5) CMA-ES (multi-start внутри, поинстансно) на лучшем.
  6) L-BFGS на итоговом лучшем.
  7) RATCHET в brain (только улучшения) + МОНОТОННЫЙ CSV + tail-отчёт
     (худшие 10 инстансов — видно, что держит среднее).

Запуск (GPU, ~30 мин/раунд по умолчанию):
  python deepgrind.py --h h_train.npy --out deep.csv --rounds 48
  # «1000 сидов за час»:  --seeds 1000 --screen-steps 15
  # только скрининг (без дорогой доводки): --fine-steps 0 --cma-gens 0 --lbfgs 0
Прерывать/перезапускать можно в любой момент (мозг + CSV монотонны).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA, ROOT  # noqa: E402
from brain import N_ANGLES, QAOABrain  # noqa: E402
from features import build_features  # noqa: E402
from model import QAOAAngleNet  # noqa: E402
from QAOA import QAOA, P  # noqa: E402
from megabrute import polish_cmaes, polish_lbfgs, split10, join10  # noqa: E402
from lucky import lbfgs_parallel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def adam_pass(qaoa, ht, a10, steps, lr, fine, fine_lr, tag):
    """Adam-проход по матрице углов (M, 10). Возвращает (a10(M,10), p(M,))."""
    g, b = split10(a10)
    g = g.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([g, b], lr=lr)
    t0 = time.time()
    for s in range(steps):
        opt.zero_grad()
        loss = -qaoa.p_ground(ht, g, b).mean()
        loss.backward()
        opt.step()
    if fine:
        opt = torch.optim.Adam([g, b], lr=fine_lr)
        for s in range(fine):
            opt.zero_grad()
            loss = -qaoa.p_ground(ht, g, b).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        p = qaoa.p_ground(ht, g, b)
    print(f"  {tag}: mean P(ground) = {p.mean().item():.4f} "
          f"({time.time() - t0:.0f} c)", flush=True)
    return join10(g.detach(), b.detach()), p.cpu().numpy()


def make_seeds(rng, B, base10, knn10, top10, n_seed):
    """n_seed матриц (B, 10): случайные, kicks, kNN+шум, топ+шум."""
    seeds = []
    n_rand = n_seed // 4
    n_kick = n_seed // 4
    n_knn = max(n_seed // 8, 1)
    n_top = max(n_seed // 8, 1)
    n_rand = max(n_rand, n_seed - n_kick - n_knn - n_top)
    for _ in range(n_rand):
        s = np.stack([rng.uniform(0, 2 * np.pi, N_ANGLES)
                      for _ in range(B)])
        s[:, P:] %= np.pi
        seeds.append(s)
    for j in range(n_kick):
        sig = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0][j % 6]
        seeds.append(base10 + rng.normal(0, sig, (B, N_ANGLES)))
    for _ in range(n_knn):
        seeds.append(knn10 + rng.normal(0, 0.4, (B, N_ANGLES)))
    for _ in range(n_top):
        seeds.append(top10 + rng.normal(0, 0.4, (B, N_ANGLES)))
    return seeds


def round_grind(qaoa, ht, h, brain, net10, cfg, seed):
    B = h.shape[0]
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # ---------- текущие лучшие (мозг / сеть) ----------
    base10 = np.empty((B, N_ANGLES))
    for i in range(B):
        k = brain.best_for(h[i])
        base10[i] = k if k is not None else net10[i]
    with torch.no_grad():
        p_base = qaoa.p_ground(ht, torch.tensor(base10[:, :P],
                                dtype=torch.float32, device=DEVICE),
                               torch.tensor(base10[:, P:],
                                dtype=torch.float32, device=DEVICE)
                               ).cpu().numpy()
    knn10 = np.empty((B, N_ANGLES))
    for i in range(B):
        kk = brain.knn_transfer(h[i])
        knn10[i] = kk if kk is not None else net10[i]
    if brain.top_schemes:
        top = brain.top_schemes[int(rng.integers(len(brain.top_schemes)))]
        top10 = np.broadcast_to(top, (B, N_ANGLES))
    else:
        top10 = net10

    # ---------- сиды (+фокус на хвост: полный доп. набор) ----------
    n_seed = cfg["seeds"]
    focus = p_base < cfg["focus"]
    print(f"── ГРУНТ: {n_seed} сидов/инстанс, фокус P<{cfg['focus']} "
          f"на {focus.sum()} инст. (+{n_seed} доп.) ──", flush=True)
    seeds_all = make_seeds(rng, B, base10, knn10, top10, n_seed)
    seeds_focus = []
    if focus.sum() > 0:
        fj = int(focus.sum())
        seeds_focus = make_seeds(rng, fj, base10[focus], knn10[focus],
                                 top10[focus], n_seed)

    # ---------- 1) COARSE скрининг ----------
    p_all = []     # (S, B) — -inf для «не применимо»
    a_all = []     # (S, B, 10)
    def run_chunk(sc, hmask=None):
        for s in sc:
            h_use = ht[hmask] if hmask is not None else ht
            a10, p = adam_pass(qaoa, h_use, s, cfg["screen_steps"], 0.05,
                               0, 0,
                               f"скрининг {len(p_all) + 1}/"
                               f"{len(seeds_all) + len(seeds_focus)}")
            if hmask is None:
                p_all.append(p)
                a_all.append(a10)
            else:
                pf = np.full(B, -np.inf)
                pf[hmask] = p
                af = np.zeros((B, N_ANGLES))
                af[hmask] = a10
                p_all.append(pf)
                a_all.append(af)
    for c0 in range(0, len(seeds_all), 8):
        run_chunk(seeds_all[c0:c0 + 8])
    for c0 in range(0, len(seeds_focus), 8):
        run_chunk(seeds_focus[c0:c0 + 8], hmask=focus)
    p_all = np.stack(p_all)
    a_all = np.stack(a_all)
    print(f"  COARSE: {p_all.shape[0]} сидов x {B} инст., "
          f"best-of = {p_all.max(axis=0).mean():.4f}", flush=True)

    # ---------- 2) top-k выживших + лучшая схема каждого инстанса ----------
    S_tot = p_all.shape[0]
    am_c = p_all.argmax(axis=0)
    p_cmax = p_all.max(axis=0)
    a_cmax = np.stack([a_all[am_c[i], i] for i in range(B)])
    best10, best_p = base10.copy(), p_base.copy()
    upd = p_cmax > best_p
    best10[upd] = a_cmax[upd]
    best_p = np.maximum(best_p, p_cmax)
    print(f"  best after COARSE: {best_p.mean():.4f}", flush=True)

    K = min(cfg["top_k"], S_tot)
    ord_s = np.argsort(-p_all, axis=0)   # (S, B)
    surv = np.empty((K, B, N_ANGLES))
    surv_p = np.empty((K, B))
    for k in range(K):
        surv_p[k] = p_all[ord_s[k], np.arange(B)]
        for i in range(B):
            surv[k, i] = a_all[ord_s[k, i], i]

    # ---------- 3) FINE (выжившие: K схем на каждый инстанс) ----------
    if cfg["fine_steps"] > 0:
        ht_f = torch.cat([ht] * K, dim=0)   # (K*B, 12) в порядке (k, i)
        a10f, pf = adam_pass(qaoa, ht_f, surv.reshape(-1, N_ANGLES),
                             cfg["fine_steps"], 0.05, cfg["fine"], 0.01,
                             f"FINE top-{K}")
        pf = pf.reshape(K, B)
        a10f = a10f.reshape(K, B, N_ANGLES)
        am = pf.argmax(axis=0)
        pf_max = pf.max(axis=0)
        upd = pf_max > best_p
        best10[upd] = a10f[am[upd], np.where(upd)[0]]
        best_p = np.maximum(best_p, pf_max)
        print(f"  best after FINE: {best_p.mean():.4f}", flush=True)

    # ---------- 4) CMA-ES (поинстансно, multi-start внутри) ----------
    if cfg["cma_gens"] > 0:
        print("  CMA-ES (глубокий, поинстансно)...", flush=True)
        out = polish_cmaes(qaoa, ht, best10.copy(), cfg["cma_gens"], seed + 7)
        with torch.no_grad():
            pc = qaoa.p_ground(ht, torch.tensor(out[:, :P],
                                 dtype=torch.float32, device=DEVICE),
                               torch.tensor(out[:, P:],
                                 dtype=torch.float32, device=DEVICE)
                               ).cpu().numpy()
        upd = pc > best_p
        best10[upd] = out[upd]
        best_p = np.maximum(best_p, pc)
        print(f"  best after CMA-ES: {best_p.mean():.4f}", flush=True)

    # ---------- 5) L-BFGS (параллельно на ядрах CPU) ----------
    if cfg["lbfgs"] > 0:
        print(f"  L-BFGS {cfg['lbfgs']} итер., {cfg['procs']} proc...",
              flush=True)
        res = lbfgs_parallel(h, best10, np.arange(B), cfg["lbfgs"],
                             cfg["j_path"], cfg["procs"], "L-BFGS")
        n_imp = 0
        for i in range(B):
            out, pn = res[i]
            if pn > best_p[i] + 1e-12:
                best10[i], best_p[i] = out, pn
                n_imp += 1
        print(f"  best after L-BFGS: {best_p.mean():.4f} "
              f"(улучшено {n_imp}/{B})", flush=True)

    # ---------- 6) ratchet в мозг + монотонный CSV ----------
    # страховка честности: пересчитать P(ground) итоговых углей напрямую
    with torch.no_grad():
        p_re = qaoa.p_ground(ht, torch.tensor(best10[:, :P],
                                 dtype=torch.float32, device=DEVICE),
                             torch.tensor(best10[:, P:],
                                 dtype=torch.float32, device=DEVICE)
                             ).cpu().numpy()
    if abs(p_re.mean() - best_p.mean()) > 5e-4:
        print(f"  !! расхождение best_p {best_p.mean():.4f} vs "
              f"пересчёт {p_re.mean():.4f} — беру пересчёт", flush=True)
    best_p = p_re
    brain.update_from_results(
        [(h[i], best10[i], best_p[i]) for i in range(B)])
    all10 = np.empty((B, N_ANGLES))
    for i in range(B):
        a = brain.best_for(h[i])
        all10[i] = a if a is not None else best10[i]
    with torch.no_grad():
        ph = qaoa.p_ground(ht, torch.tensor(all10[:, :P],
                                dtype=torch.float32, device=DEVICE),
                           torch.tensor(all10[:, P:],
                                dtype=torch.float32, device=DEVICE)
                           ).cpu().numpy()
    improved = int((best_p > p_base + 1e-12).sum())
    tail = np.argsort(ph)[:10]
    print(f"  улучшено: {improved}/{B} | этот раунд: {best_p.mean():.4f} "
          f"| ИСТОРИЯ: {ph.mean():.4f} ({time.time() - t0:.0f} c)",
          flush=True)
    print("  tail (худшие 10 — держат среднее): " +
          " ".join(f"#{i}:{ph[i]:.3f}" for i in tail), flush=True)
    return all10, ph.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="h_train.npy")
    ap.add_argument("--out", default="deep.csv")
    ap.add_argument("--rounds", type=int, default=48)
    ap.add_argument("--start-seed", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=48,
                    help="сидов на инстанс (1000 = «тысяча за час»)")
    ap.add_argument("--screen-steps", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--fine-steps", type=int, default=300)
    ap.add_argument("--fine", type=int, default=100)
    ap.add_argument("--cma-gens", type=int, default=80)
    ap.add_argument("--lbfgs", type=int, default=100)
    ap.add_argument("--focus", type=float, default=0.6,
                    help="инстансы с P ниже получают +полный набор сидов")
    ap.add_argument("--procs", type=int, default=4,
                    help="ядер CPU для параллельного L-BFGS (1 = по одному)")
    ap.add_argument("--min-improve", type=float, default=0.0002)
    args = ap.parse_args()

    cfg = dict(seeds=args.seeds, screen_steps=args.screen_steps,
               top_k=args.top_k, fine_steps=args.fine_steps,
               fine=args.fine, cma_gens=args.cma_gens,
               lbfgs=args.lbfgs,
               focus=args.focus if args.focus > 0 else 1e9,
               procs=max(args.procs, 1),
               j_path=os.path.join(ROOT, "J.npy"))

    h_path = args.h if os.path.exists(args.h) else os.path.join(ROOT, args.h)
    h = np.load(h_path)
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
    with torch.no_grad():
        g0, b0 = net.angles(xt)
    net10 = np.concatenate([g0.cpu().numpy(), b0.cpu().numpy()], axis=1)

    brain_path = os.path.join(DATA, "brain.pkl")
    brain = QAOABrain.load(brain_path)
    print(f"device: {DEVICE}, N = {len(h)}, мозг: {len(brain)} схем, "
          f"раундов: {args.rounds}", flush=True)

    cols = (["id"] + [f"gamma_{k}" for k in range(P)]
            + [f"beta_{k}" for k in range(P)])
    hist_best = -1.0
    stagnant = 0
    for r in range(args.rounds):
        seed = args.start_seed + r
        print(f"\n########## ГРУНТ {r + 1}/{args.rounds} (seed {seed}) "
              f"##########", flush=True)
        all10, phist = round_grind(qaoa, ht, h, brain, net10, cfg, seed)
        out = np.concatenate([np.arange(len(h))[:, None], all10], axis=1)
        np.savetxt(args.out, out, delimiter=",", header=",".join(cols),
                   comments="", fmt=["%d"] + ["%.12f"] * 10)
        brain.save(brain_path)
        if phist <= hist_best + args.min_improve:
            stagnant += 1
        else:
            stagnant = 0
        hist_best = max(hist_best, phist)
        if stagnant >= 5 and args.rounds > 1:
            print(f"плато: история не растёт 5 раундов — стоп. "
                  f"Лучшее: {hist_best:.4f}", flush=True)
            break
    print(f"\nГОТОВО. Итог (история): {hist_best:.4f}, файл: {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
