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
  full  — для финальной посылки (GPU): 8 рестартов x 300 шагов + доводка;
  fast  — быстрая самопроверка на CPU: 3 рестарта x 100 шагов;
  brain — популяционный поиск с персистентным «мозгом» (аналог подхода
          MCTech: умная инициализация из памяти, отбор, скрещивание,
          мутация, L-BFGS-доводка). 8 кандидатов x 2 поколения x 150
          шагов + доводка; память — data/brain.pkl.

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
from brain import N_ANGLES, QAOABrain  # noqa: E402
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


def polish_brain(qaoa, ht, h_np, g0, b0, brain, pop, gens, steps, fine_steps,
                 fine_lr, lr, refine=True, refine_iters=25, seed=42):
    """Поинстансный популяционный поиск — аналог ГА из MCTech:

      популяция угловых схем (умная инициализация из «мозга»)
        -> «симулятор» (Adam-шаги QAOA) -> отбор по P(ground)
        -> скрещивание / мутация -> следующие поколения
        -> финальная доводка (Adam, мелкий lr) + L-BFGS-доводка.

    Для каждого инстанса остаётся лучшая схема; всё лучшее передаётся
    в «мозг» (см. QAOABrain.update_from_results).
    """
    B = ht.shape[0]
    dev = ht.device
    rng = np.random.default_rng(seed)
    g0n, b0n = g0.cpu().numpy(), b0.cpu().numpy()
    net10 = np.concatenate([g0n, b0n], axis=1)

    def split(a10):
        a = torch.tensor(a10, dtype=torch.float32, device=dev)
        return a[:, :P], a[:, P:]

    # ---------- начальная популяция («умная инициализация») ----------
    row1 = (np.concatenate([g0n, b0n], axis=1)
            + rng.normal(0.0, 0.3, (B, N_ANGLES)))
    n_known = 0
    for i in range(B):
        k = brain.best_for(h_np[i])
        if k is not None:           # этот h мы уже решали — берём из памяти
            row1[i] = k
            n_known += 1
    # kNN-трансфер: углы ближайших по h схем из памяти (parameter transfer)
    knn_row = np.empty((B, N_ANGLES))
    for i in range(B):
        k = brain.knn_transfer(h_np[i])
        knn_row[i] = (k if k is not None
                      else brain.smart_randomize(rng, net10[i]))
    # PCA-многообразие хороших углов (QAOA-PCA): мутации по главным осям
    man_row = np.empty((B, N_ANGLES))
    for i in range(B):
        m = brain.sample_manifold(rng)
        man_row[i] = (m if m is not None
                      else brain.smart_randomize(rng, net10[i]))
    pop10 = [net10.copy(), row1, knn_row, man_row]
    for _ in range(pop - len(pop10)):
        pop10.append(np.stack(
            [brain.smart_randomize(rng, net10[i]) for i in range(B)]))
    if n_known:
        print(f"  из памяти «мозга» взяты схемы для {n_known} инстансов",
              flush=True)

    # ---------- эволюция ----------
    best10, best_fit = None, None
    for gen in range(gens):
        fit = np.zeros((len(pop10), B))
        opt10 = [None] * len(pop10)
        for c in range(len(pop10)):
            g, b = split(pop10[c])
            gs, bs, p = run_restart(
                qaoa, ht, g, b, steps, lr, 0, 0,
                f"поколение {gen + 1}/{gens}, кандидат {c + 1}/{len(pop10)}")
            fit[c] = p.cpu().numpy()
            # в популяцию следующего поколения идёт УЖЕ ОПТИМИЗИРОВАННАЯ схема
            opt10[c] = np.concatenate([gs.cpu().numpy(),
                                       bs.cpu().numpy()], axis=1)
        am = fit.argmax(axis=0)
        cur10 = np.stack([opt10[am[i]][i] for i in range(B)])
        cur_fit = fit[am, np.arange(B)]
        if best10 is None:
            best10, best_fit = cur10, cur_fit
        else:
            upd = cur_fit > best_fit
            best10 = np.where(upd[:, None], cur10, best10)
            best_fit = np.maximum(best_fit, cur_fit)
        print(f"  -> best-of mean P(ground) = {best_fit.mean():.4f} "
              f"(поинстансный выбор из {len(pop10)} кандидатов)", flush=True)
        # ВИДИМОСТЬ: best-of — ПО КАЖДОМУ инстансу, а не среднее из
        # средних: каждый кандидат побеждает на СВОЕЙ подгруппе инстансов,
        # поэтому итог выше максимального среднего кандидата
        win = np.bincount(am, minlength=len(pop10))
        parts = [f"к{c + 1}:{win[c]}"
                 + (f"(там {fit[c][am == c].mean():.3f})" if win[c] else "")
                 for c in range(len(pop10))]
        print("     победил поинстансно: " + " | ".join(parts), flush=True)
        order = np.argsort(-fit, axis=0)  # (n_cand, B): order[ранг, инстанс]
        t1 = np.stack([opt10[c][i] for i, c in enumerate(order[0])])
        t2 = np.stack([opt10[c][i] for i, c in enumerate(order[1])])
        nxt = [t1, t2]                     # элитизм: лучшие выживают
        # DE/rand/1/bin (arXiv 2303.12186): мутация из РАЗНОСТЕЙ схем популяции
        # + бинарное скрещивание — выталкивает из локальных минимумов лучше,
        # чем гауссов шум
        F, CR = 0.8, 0.5
        r1 = np.stack([brain.smart_randomize(rng, net10[i]) for i in range(B)])
        r2 = np.stack([brain.smart_randomize(rng, net10[i]) for i in range(B)])
        mask = rng.random((B, N_ANGLES)) < CR
        mask[:, 0] = True
        nxt.append(np.where(mask, t1 + F * (t2 - r1), t2))
        mask = rng.random((B, N_ANGLES)) < CR
        mask[:, 0] = True
        nxt.append(np.where(mask, t2 + F * (t1 - r2), t1))
        while len(nxt) < pop:
            nxt.append(np.stack(
                [brain.smart_randomize(rng, net10[i]) for i in range(B)]))
        pop10 = nxt[:pop]

    # ---------- финальная доводка лучшего ----------
    g, b = split(best10)
    g, b, _ = run_restart(qaoa, ht, g, b, 0, lr, fine_steps, fine_lr,
                          "доводка (fine)")

    # ---------- L-BFGS-доводка («final refinement» из MCTech) ----------
    if refine:
        g_np, b_np = g.cpu().numpy(), b.cpu().numpy()
        t0 = time.time()
        for i in range(B):
            gi = torch.tensor(g_np[i], dtype=torch.float32, device=dev,
                              requires_grad=True)
            bi = torch.tensor(b_np[i], dtype=torch.float32, device=dev,
                              requires_grad=True)

            def closure():
                opt.zero_grad()
                loss = -qaoa.p_ground(ht[i:i + 1],
                                      gi.unsqueeze(0), bi.unsqueeze(0)).mean()
                loss.backward()
                return loss

            opt = torch.optim.LBFGS([gi, bi], lr=1.0, max_iter=refine_iters,
                                    history_size=10)
            opt.step(closure)
            g_np[i] = gi.detach().cpu().numpy()
            b_np[i] = bi.detach().cpu().numpy()
        g, b = (torch.tensor(g_np, device=dev), torch.tensor(b_np, device=dev))
        print(f"  L-BFGS-доводка: {time.time() - t0:.0f} c", flush=True)
    return g, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="h_test.npy", help="файл с h")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--profile", choices=["full", "fast", "brain"],
                    default="full",
                    help="full — финальная (GPU), fast — самопроверка (CPU),\n"
                         "brain — популяционный поиск с «мозгом» (MCTech-подход)")
    ap.add_argument("--pop", type=int, default=8,
                    help="размер популяции (только профиль brain)")
    ap.add_argument("--gens", type=int, default=2,
                    help="число поколений (только профиль brain)")
    ap.add_argument("--no-refine", action="store_true",
                    help="выключить L-BFGS-доводку (только профиль brain)")
    ap.add_argument("--refine-iters", type=int, default=25,
                    help="итераций L-BFGS на инстанс (только профиль brain)")
    ap.add_argument("--restarts", type=int, default=None,
                    help="переопределить число рестартов (по умолчанию — из профиля)")
    ap.add_argument("--steps", type=int, default=None,
                    help="переопределить число шагов Adam на рестарт")
    ap.add_argument("--fine", type=int, default=None,
                    help="переопределить число шагов финальной доводки")
    ap.add_argument("--limit", type=int, default=0,
                    help="обработать только первые N (только для теста)")
    ap.add_argument("--seed", type=int, default=42,
                    help="сид для случайных рестартов/популяции")
    args = ap.parse_args()

    if args.profile == "full":
        R, S, FS = POLISH_RESTARTS, POLISH_STEPS, POLISH_FINE_STEPS
    elif args.profile == "fast":
        R, S, FS = (POLISH_RESTARTS_FAST, POLISH_STEPS_FAST,
                    POLISH_FINE_STEPS_FAST)
    else:  # brain: 8 кандидатов x 2 поколения x 150 шагов + доводка
        R, S, FS = None, 150, POLISH_FINE_STEPS
    R = args.restarts if args.restarts is not None else R
    S = args.steps if args.steps is not None else S
    FS = args.fine if args.fine is not None else FS
    if args.profile == "brain":
        print(f"profile=brain: pop={args.pop}, gens={args.gens}, "
              f"steps={S}, fine={FS}, refine={not args.no_refine}")
    else:
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
    brain = None
    if args.profile == "brain":
        brain_path = os.path.join(DATA, "brain.pkl")
        brain = QAOABrain.load(brain_path)
        print(f"мозг: {len(brain)} схем из {brain_path}"
              + ("" if len(brain) else " (пустой — первый прогон)"))
        g, b = polish_brain(qaoa, ht, h, g0, b0, brain,
                            args.pop, args.gens, S, FS,
                            POLISH_FINE_LR, POLISH_LR,
                            refine=not args.no_refine,
                            refine_iters=args.refine_iters,
                            seed=args.seed)
    else:
        g, b = polish_strong(qaoa, ht, g0, b0, R, S, FS, POLISH_FINE_LR,
                             POLISH_LR, seed=args.seed)
    dt = time.time() - t_total

    with torch.no_grad():
        p_after = qaoa.p_ground(ht, g, b).mean().item()
    print(f"\nP(ground) после полировки: {p_after:.4f}  (сеть давала {p_before:.4f})")
    print(f"время полировки: {dt:.1f} c (лимит 600 c)")

    g_np, b_np = g.cpu().numpy(), b.cpu().numpy()
    assert len(np.unique(g_np, axis=0)) > 1, "углы должны зависеть от h"
    cols = (["id"] + [f"gamma_{k}" for k in range(5)]
            + [f"beta_{k}" for k in range(5)])

    if brain is not None:
        with torch.no_grad():
            p_round = qaoa.p_ground(ht, g, b)
        brain.update_from_results(
            [(h[i], np.concatenate([g_np[i], b_np[i]]), p_round[i].item())
             for i in range(len(h))])
        brain.save(os.path.join(DATA, "brain.pkl"))
        # СТРОГО МОНОТОННЫЙ CSV: для каждого инстанса — лучший из ВСЕХ,
        # что когда-либо найдены (мозг хранит только улучшения)
        best10 = np.empty((len(h), N_ANGLES))
        for i in range(len(h)):
            a = brain.best_for(h[i])
            best10[i] = (a if a is not None
                         else np.concatenate([g_np[i], b_np[i]]))
        gb = torch.tensor(best10[:, :P], dtype=torch.float32,
                          device=ht.device)
        bb = torch.tensor(best10[:, P:], dtype=torch.float32,
                          device=ht.device)
        with torch.no_grad():
            p_hist = qaoa.p_ground(ht, gb, bb)
        out = np.concatenate([np.arange(len(h))[:, None], best10], axis=1)
        np.savetxt(args.out, out, delimiter=",",
                   header=",".join(cols), comments="", fmt=["%d"] + ["%.12f"] * 10)
        print(f"сохранено (ИТОГОВЫЙ BEST OF ALL ROUNDS): {args.out}")
        print(f"этот раунд: {p_round.mean().item():.4f} | "
              f"лучшее за всю историю: {p_hist.mean().item():.4f}")
        print(f"мозг обновлён: {len(brain)} схем -> "
              f"{os.path.join(DATA, 'brain.pkl')}")
    else:
        out = np.concatenate(
            [np.arange(len(h))[:, None], g_np, b_np], axis=1)
        np.savetxt(args.out, out, delimiter=",",
                   header=",".join(cols), comments="", fmt=["%d"] + ["%.12f"] * 10)
        print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
