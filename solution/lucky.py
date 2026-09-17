"""LUCKY — быстрые «охотничьи» проходы: максимум удачи за минимум времени.

Идея: среднее держит ХВОСТ (худшие 80-150 инстансов). Каждый проход:
  1) берём худшие --hunt инстансы (по текущему CSV);
  2) бросаем в них --kicks KICKS текущего лучшего угла (σ = 0.3..3.0,
     лог-равномерно) — каждый kick дожимаем Adam --steps (батчами по 8
     kicks на GPU);
  3) per-instance RATCHET (никогда не хужее входного CSV);
  4) COORD-доводка (--coord N): N раундов поочерёдного поиска —
     каждый из 10 углов перебирается по всему кругу 2π (32 точки),
     подхватывает узкие пики, которые шаг Adam перескочил;
  5) L-BFGS --lbfgs итераций параллельно на --procs ядрах CPU;
  6) ratchet + запись CSV + tail-отчёт.

~10-15 мин/проход на GPU. Гонять сериями: --rounds 12 (~2 ч).
НЕ трогает brain.pkl/model.pt → можно гонять ПАРаллельно с megabrute
(если VRAM не хватает — поочерёдно).

Запуск:
  python lucky.py --h h_train.npy --from mega.csv --out lucky.csv --rounds 12
  # агрессивнее: --hunt 120 --kicks 200 --steps 200 --procs 6
  # всё сразу (без хвоста): --hunt 0
  # ДЕНЬГИ В СЕРЕДИНЕ (поимки идут в полосе 0.1-0.3, мёртвый хвост 0.04-0.06
  # за тысячи kicks не двигается): --band-low 0.08 --band-high 0.30
  # 'телепорт' для мёртвых: --hunt 15 --sigma-max 8 (почти равномерный сброс)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QAOA import QAOA, FastQAOA, P  # noqa: E402
from brain import N_ANGLES  # noqa: E402
from megabrute import split10, join10  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_LBFGS_CACHE = {}


def _lbfgs_task(task):
    """L-BFGS на ОДНОМ инстансе. Топ-уровень: воркер ProcessPool."""
    idx, h_row, a_row, iters, j_path = task
    torch.set_num_threads(1)
    if "q" not in _LBFGS_CACHE:
        _LBFGS_CACHE["q"] = QAOA(np.load(j_path),
                                 device=DEVICE)
    q = _LBFGS_CACHE["q"]
    ht = torch.tensor(h_row[None, :], dtype=torch.float32, device=DEVICE)
    g = torch.tensor(a_row[:P], dtype=torch.float32, device=DEVICE,
                     requires_grad=True)
    b = torch.tensor(a_row[P:], dtype=torch.float32, device=DEVICE,
                     requires_grad=True)

    def closure():
        opt.zero_grad()
        loss = -q.p_ground(ht, g.unsqueeze(0), b.unsqueeze(0)).mean()
        loss.backward()
        return loss

    opt = torch.optim.LBFGS([g, b], lr=1.0, max_iter=iters,
                            history_size=10)
    opt.step(closure)
    out = np.concatenate([g.detach().cpu().numpy(),
                          b.detach().cpu().numpy()])
    with torch.no_grad():
        p = q.p_ground(ht, g.unsqueeze(0), b.unsqueeze(0)).item()
    return idx, out, p


def lbfgs_parallel(h, a10, idx, iters, j_path, procs, tag):
    """L-BFGS на инстансах idx (матрица a10). Возвращает {idx: (a, p)}."""
    tasks = [(int(i), h[i], a10[i], iters, j_path) for i in idx]
    res = {}
    step = max(len(tasks) // 10, 1)
    try:
        if procs <= 1:
            for k, t in enumerate(tasks):
                i, out, p = _lbfgs_task(t)
                res[i] = (out, p)
                if k % step == 0:
                    print(f"  {tag} {k + 1}/{len(tasks)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=procs) as ex:
                for k, (i, out, p) in enumerate(
                        ex.map(_lbfgs_task, tasks, chunksize=8)):
                    res[i] = (out, p)
                    if k % step == 0:
                        print(f"  {tag} {k + 1}/{len(tasks)}", flush=True)
    except Exception as e:  # fallback: параллелизм сломался — по одному
        print(f"  !! параллельный L-BFGS сломался ({type(e).__name__}: "
              f"{e}) — по одному", flush=True)
        res = {}
        for k, t in enumerate(tasks):
            i, out, p = _lbfgs_task(t)
            res[i] = (out, p)
            if k % step == 0:
                print(f"  {tag} {k + 1}/{len(tasks)}", flush=True)
    print(flush=True)
    return res


def adam_pass(qaoa, ht, a10, steps, lr, tag):
    g, b = split10(a10)
    g = g.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([g, b], lr=lr)
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad()
        loss = -qaoa.p_ground(ht, g, b).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = qaoa.p_ground(ht, g, b)
    print(f"  {tag}: mean P = {p.mean().item():.4f} "
          f"({time.time() - t0:.0f} c)", flush=True)
    return join10(g.detach(), b.detach()), p.cpu().numpy()


def p_eval(qaoa, ht, a10):
    g, b = split10(a10)
    with torch.no_grad():
        return qaoa.p_ground(ht, g, b).cpu().numpy()


def informed_angles(Jn, H, rng):
    """beta-informed init (A/B +41% у конкурентов): γ,β ~ U(0, 0.6),
    1-3 случайных слоя со смещением β к π/2 (+N(0, 0.15))."""
    a = rng.uniform(0.0, 0.6, (Jn, H, N_ANGLES))
    for k in range(Jn):
        for _ in range(int(rng.integers(1, 4))):
            layer = int(rng.integers(0, P))
            a[k, :, P + layer] = np.pi / 2 + rng.normal(0.0, 0.15, H)
    return a


def hunt_pass(qaoa, ht_sel, base, p0, kicks, steps, seed, tag, sigma_max=3.0,
              wide_frac=0.3, informed_frac=0.2):
    """kicks x Adam на H инстансах. Ratchet внутри. (a, p) >= (base, p0).
    Три семейства киков (доли задаются): local — base + σ·N (как раньше);
    wide — base × {1.5,2,3,4} (пробивает потолок hard h: победные |γ|~4.9π);
    informed — свежие узкие углы с β≈π/2 слоями."""
    H = base.shape[0]
    rng = np.random.default_rng(seed)
    sig = np.logspace(np.log10(0.3), np.log10(sigma_max), kicks)
    mults = np.array([1.5, 2.0, 3.0, 4.0])
    a_best, p_best = base.copy(), p0.copy()
    n_done, n_tot = 0, int(np.ceil(kicks / 8))
    for c0 in range(0, kicks, 8):
        Jn = min(8, kicks - c0)
        n_done += 1
        a10 = base[None] + sig[c0:c0 + Jn][:, None, None] * \
            rng.normal(0.0, 1.0, (Jn, H, N_ANGLES))
        r = rng.random(Jn)
        wide = r < wide_frac
        inf = (r >= wide_frac) & (r < wide_frac + informed_frac)
        if wide.any():
            mw = mults[(c0 + np.arange(Jn)) % 4][wide][:, None, None]
            a10[wide] = base[None] * mw + rng.normal(0.0, 0.05, a10[wide].shape)
        if inf.any():
            a10[inf] = informed_angles(int(inf.sum()), H, rng)
        ht_c = torch.cat([ht_sel] * Jn, dim=0)
        a10n, pn = adam_pass(qaoa, ht_c, a10.reshape(-1, N_ANGLES),
                             steps, 0.05,
                             f"{tag} kicks {c0 + 1}-{c0 + Jn}/"
                             f"{kicks} [{n_done}/{n_tot}]")
        pn = pn.reshape(Jn, H)
        a10n = a10n.reshape(Jn, H, N_ANGLES)
        am = pn.argmax(axis=0)
        pn_max = pn.max(axis=0)
        upd = pn_max > p_best
        a_best[upd] = a10n[am[upd], np.where(upd)[0]]
        p_best[upd] = pn_max[upd]
    return a_best, p_best


def save_csv(path, a, ids):
    cols = (["id"] + [f"gamma_{k}" for k in range(P)]
            + [f"beta_{k}" for k in range(P)])
    out = np.concatenate([np.asarray(ids)[:, None], a], axis=1)
    np.savetxt(path, out, delimiter=",", header=",".join(cols),
               comments="", fmt=["%d"] + ["%.12f"] * N_ANGLES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="h_train.npy")
    ap.add_argument("--from", dest="src", default="mega.csv",
                    help="CSV с текущими лучшими углами")
    ap.add_argument("--out", default="lucky.csv")
    ap.add_argument("--hunt", type=int, default=80,
                    help="охотимся на худшие N инстансов (0 = все)")
    ap.add_argument("--kicks", type=int, default=120)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lbfgs", type=int, default=50)
    ap.add_argument("--coord", type=int, default=0,
                    help="N раундов поочерённого (координатного) дожима: "
                         "каждый угол по всему кругу 2π, 32 точки. 0 = выкл")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--alert", type=float, default=0.05,
                    help="печатать инстансы, прыгнувшие больше этого за проход")
    ap.add_argument("--cross", type=float, default=0.5,
                    help="печатать инстансы, пересекшие этот порог (0.5/0.7/0.9)")
    ap.add_argument("--wide-frac", dest="wide_frac", type=float, default=0.3,
                    help="доля wide-киков (base x 1.5/2/3/4), 0=выкл")
    ap.add_argument("--informed-frac", dest="informed_frac", type=float, default=0.2,
                    help="доля informed-киков (узкие + beta~pi/2), 0=выкл")
    ap.add_argument("--sigma-max", dest="sigma_max", type=float, default=3.0,
                    help="максимальный σ kick. 3 = стандарт; 8-10 = 'телепорт' "
                         "для мёртвых инстансов (почти равномерный сброс)")
    ap.add_argument("--band-low", dest="band_low", type=float, default=0.0,
                    help="охотиться только на инстансы с P >= этого "
                         "(0.08 = пропустить мёртвый хвост)")
    ap.add_argument("--band-high", dest="band_high", type=float, default=10.0,
                    help="... и P < этого (0.30 = не трогать уже хорошие)")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h_path = args.h if os.path.exists(args.h) else os.path.join(root, args.h)
    src_path = (args.src if os.path.exists(args.src)
                else os.path.join(root, args.src))
    h = np.load(h_path)
    j_path = os.path.join(root, "J.npy")
    qaoa = FastQAOA(np.load(j_path), device=DEVICE)  # ~6x быстрее QAOA
    ht = torch.tensor(h, dtype=torch.float32, device=DEVICE)

    data = np.genfromtxt(src_path, delimiter=",", skip_header=1)
    ids, a = data[:, 0].astype(int), data[:, 1:11].astype(np.float32)
    assert a.shape[0] == h.shape[0], f"CSV {a.shape[0]} != h {h.shape[0]}"
    p = p_eval(qaoa, ht, a)
    p0_mean = p.mean()
    p0_full = p.copy()  # для «топ скачков за сессию» в конце
    print(f"старт: mean P(ground) = {p0_mean:.4f} (из {os.path.basename(args.src)})",
          flush=True)
    band_s = ("" if (args.band_low <= 0 and args.band_high >= 1)
              else f", полоса P {args.band_low:g}..{args.band_high:g}")
    print(f"охота: худшие {args.hunt if args.hunt > 0 else 'ВСЕ'} x "
          f"{args.kicks} kicks (σ 0.3..{args.sigma_max:g}; wide {args.wide_frac:g}, "
          f"informed {args.informed_frac:g}) x Adam {args.steps} "
          f"+ L-BFGS {args.lbfgs} ({args.procs} proc), проходов: "
          f"{args.rounds}{band_s}", flush=True)

    for r in range(args.rounds):
        if args.hunt > 0:
            # band: сначала худшие в полосе [band_low, band_high),
            # если в полосе меньше --hunt — добираем худшими вне полосы
            in_band = (p >= args.band_low) & (p < args.band_high)
            order = np.argsort(p)  # худшие первыми
            band_order = order[in_band[order]]
            out_order = order[~in_band[order]]
            sel_list = list(band_order)
            for i in out_order:
                if len(sel_list) >= args.hunt:
                    break
                sel_list.append(i)
            sel = np.array(sel_list[:args.hunt])
        else:
            sel = np.arange(h.shape[0])
        print(f"\n=== ПРОХОД {r + 1}/{args.rounds}: худшие {len(sel)} "
              f"(P {p[sel].min():.3f}..{p[sel].max():.3f}) ===", flush=True)
        t0 = time.time()
        p_pass_start = p.copy()
        ht_sel = ht[sel]
        a_sel, p_sel = hunt_pass(qaoa, ht_sel, a[sel], p[sel], args.kicks,
                                 args.steps, args.seed + r,
                                 f"проход {r + 1}", sigma_max=args.sigma_max,
                                 wide_frac=args.wide_frac,
                                 informed_frac=args.informed_frac)
        a[sel], p[sel] = a_sel, p_sel
        if args.coord > 0:
            from coord import coord_pass  # lazy: круг lucky<->coord
            for r2 in range(args.coord):
                a_sel, p_sel = coord_pass(qaoa, ht_sel, a[sel], p[sel], 32,
                                          f"проход {r + 1} coord "
                                          f"{r2 + 1}/{args.coord}")
                a[sel], p[sel] = a_sel, p_sel
        if args.lbfgs > 0:
            res = lbfgs_parallel(h, a, sel, args.lbfgs, j_path, args.procs,
                                 f"L-BFGS проход {r + 1}")
            n_imp = 0
            for i in sel:
                out, pn = res[int(i)]
                if pn > p[i] + 1e-12:
                    a[i], p[i] = out, pn
                    n_imp += 1
            print(f"  L-BFGS: улучшено {n_imp}/{len(sel)} | mean = "
                  f"{p.mean():.4f}", flush=True)
        # --- видимость «ПОЙМАЛИ» (вдруг поймал глобальный бассейн) ---
        gain = p - p_pass_start
        hot = np.where(gain > args.alert)[0]
        for i in sorted(hot, key=lambda i: -gain[i])[:15]:
            print(f"  ** ПОЙМАЛИ #{int(ids[i])}: "
                  f"{p_pass_start[i]:.3f} -> {p[i]:.3f} "
                  f"(+{gain[i]:.3f})", flush=True)
        crossed = np.where((p >= args.cross) & (p_pass_start < args.cross))[0]
        for i in sorted(crossed)[:15]:
            print(f"  ** ПОРОГ {args.cross} пройден: #{int(ids[i])} -> "
                  f"{p[i]:.3f}", flush=True)
        save_csv(args.out, a, ids)
        tail = np.argsort(p)[:10]
        print(f"  проход {r + 1}: mean = {p.mean():.4f} "
              f"(+{p.mean() - p0_mean:+.4f} от старта, {time.time() - t0:.0f} c)",
              flush=True)
        print("  tail: " + " ".join(f"#{i}:{p[i]:.3f}" for i in tail),
              flush=True)

    # --- топ скачков за сессию (видимость, что вообще поймали) ---
    tot_gain = p - p0_full
    top = np.argsort(-tot_gain)[:10]
    print("\nТОП-10 скачков за сессию (кто и куда прыгнул):", flush=True)
    for i in top:
        print(f"  #{int(ids[i])}: {p0_full[i]:.3f} -> {p[i]:.3f} "
              f"(+{tot_gain[i]:.3f})", flush=True)
    over = (p >= args.cross).sum()
    print(f"инстансов выше порога {args.cross}: {int(over)} из {len(p)}",
          flush=True)

    print(f"\nИТОГ: mean P(ground) = {p.mean():.4f}  (старт {p0_mean:.4f})",
          flush=True)
    print(f"гарантия: {args.out} никогда хуже {args.src} (ratchet).",
          flush=True)
    print(f"файл: {args.out}", flush=True)


if __name__ == "__main__":
    main()
