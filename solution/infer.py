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

Жёсткий бюджет времени (--time-budget, сек)
-------------------------------------------
Лимит задачи на h_test — 600 с. `--time-budget 520` включает страховку:
полировка сама измеряет стоимость рестарта/кандидата и перестаёт начинать
новые, если остаток бюджета их не покрывает; поинстансная L-BFGS-доводка
прерывается на том инстансе, где бюджет исчерпан. Углы при этом остаются
валидными (лучшие из уже полученных), submission.csv записывается всегда.
Для оффлайн-раундов на h_train бюджет не нужен (лимит там не действует) —
по умолчанию 0, т.е. без ограничения.
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

# Сколько секунд оставить в запасе при --time-budget на финальную оценку
# P(ground), обновление «мозга» и запись submission.csv.
BUDGET_RESERVE = 20.0


def budget_reserve(budget):
    """Резерв под хвост (оценка + запись CSV). Для боевого бюджета 520 с это
    20 с; для коротких тестовых бюджетов резерв урезается, иначе полировка
    останавливалась бы, не начавшись."""
    if not budget or budget <= 0:
        return BUDGET_RESERVE
    return float(min(BUDGET_RESERVE, max(2.0, 0.05 * budget)))


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
                  fine_lr, lr, seed=42, deadline=None,
                  reserve=BUDGET_RESERVE):
    """Много-рестарт поинстансная полировка, best-of по P(ground).

    deadline — момент времени (time.time()), после которого новые рестарты
    не начинаются (страховка от лимита 600 с на h_test).
    """
    B = ht.shape[0]
    torch.manual_seed(seed)
    best_g = g0.detach().clone()
    best_b = b0.detach().clone()
    best_p = -torch.ones(B, device=ht.device)
    dt_hist = []

    for r in range(restarts):
        if deadline is not None and r > 0 and dt_hist:
            est = float(np.mean(dt_hist))
            left = deadline - time.time()
            if left < est + reserve:
                print(f"  [бюджет] рестарт {r + 1}/{restarts} пропущен: "
                      f"нужно ~{est:.0f} c, осталось {left:.0f} c",
                      flush=True)
                break
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
        t_r = time.time()
        gs, bs, p = run_restart(qaoa, ht, g, b, steps, lr,
                                fine_steps, fine_lr, tag)
        dt_hist.append(time.time() - t_r)
        upd = (p > best_p).unsqueeze(1)
        best_g = torch.where(upd, gs, best_g)
        best_b = torch.where(upd, bs, best_b)
        best_p = torch.maximum(p, best_p)
        print(f"  -> best-of mean P(ground) = {best_p.mean().item():.4f}",
              flush=True)
        if deadline is not None:
            print(f"  [бюджет] осталось {deadline - time.time():.0f} c",
                  flush=True)
    return best_g, best_b


def polish_brain(qaoa, ht, h_np, g0, b0, brain, pop, gens, steps, fine_steps,
                 fine_lr, lr, refine=True, refine_iters=25, seed=42,
                 deadline=None, reserve=BUDGET_RESERVE):
    """Поинстансный популяционный поиск — аналог ГА из MCTech:

      популяция угловых схем (умная инициализация из «мозга»)
        -> «симулятор» (Adam-шаги QAOA) -> отбор по P(ground)
        -> скрещивание / мутация -> следующие поколения
        -> финальная доводка (Adam, мелкий lr) + L-BFGS-доводка.

    Для каждого инстанса остаётся лучшая схема; всё лучшее передаётся
    в «мозг» (см. QAOABrain.update_from_results).

    deadline — момент time.time(), к которому нужно успеть: новые кандидаты
    и поколения не начинаются, если остаток бюджета их не покрывает,
    L-BFGS-доводка прерывается на инстансе. Результат всегда валиден.
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
    pop10 = [net10.copy(), row1]
    for _ in range(pop - len(pop10)):
        pop10.append(np.stack(
            [brain.smart_randomize(rng, net10[i]) for i in range(B)]))
    if n_known:
        print(f"  из памяти «мозга» взяты схемы для {n_known} инстансов",
              flush=True)

    # ---------- эволюция ----------
    best10, best_fit = None, None
    cand_dt = None                       # замеренная стоимость одного кандидата
    for gen in range(gens):
        fit_rows, opt10 = [], []
        for c in range(len(pop10)):
            if deadline is not None and cand_dt is not None:
                left = deadline - time.time()
                if left < cand_dt + reserve:
                    print(f"  [бюджет] поколение {gen + 1}, кандидат {c + 1}"
                          f"/{len(pop10)} пропущен: нужно ~{cand_dt:.0f} c, "
                          f"осталось {left:.0f} c", flush=True)
                    break
            g, b = split(pop10[c])
            t_c = time.time()
            gs, bs, p = run_restart(
                qaoa, ht, g, b, steps, lr, 0, 0,
                f"поколение {gen + 1}/{gens}, кандидат {c + 1}/{len(pop10)}")
            cand_dt = time.time() - t_c
            fit_rows.append(p.cpu().numpy())
            # в популяцию следующего поколения идёт УЖЕ ОПТИМИЗИРОВАННАЯ схема
            opt10.append(np.concatenate([gs.cpu().numpy(),
                                         bs.cpu().numpy()], axis=1))
        if not fit_rows:
            print("  [бюджет] поколение не начато — эволюция остановлена",
                  flush=True)
            break
        # fit: (n_run, B) — n_run может быть меньше len(pop10) при бюджете
        fit = np.stack(fit_rows)
        am = fit.argmax(axis=0)
        cur10 = np.stack([opt10[am[i]][i] for i in range(B)])
        cur_fit = fit[am, np.arange(B)]
        if best10 is None:
            best10, best_fit = cur10, cur_fit
        else:
            upd = cur_fit > best_fit
            best10 = np.where(upd[:, None], cur10, best10)
            best_fit = np.maximum(best_fit, cur_fit)
        print(f"  -> best-of mean P(ground) = {best_fit.mean():.4f}",
              flush=True)
        if fit.shape[0] >= 2:
            # отбор: для КАЖДОГО инстанса i — лучший (order[0]) и второй
            # (order[1]) кандидат. order имеет форму (n_run, B), поэтому
            # индекс кандидата — это order[k][i], а не order[i, k].
            order = np.argsort(-fit, axis=0)
            t1 = np.stack([opt10[c][i] for i, c in enumerate(order[0])])
            t2 = np.stack([opt10[c][i] for i, c in enumerate(order[1])])
        else:                            # единственный выживший кандидат
            t1 = t2 = cur10
        nxt = [t1, t2,                       # элитизм: лучшие выживают
               brain.mutate(t1, rng, 0.15),
               brain.mutate(t2, rng, 0.15),
               brain.crossover(t1, t2, rng)]
        while len(nxt) < pop:
            nxt.append(np.stack(
                [brain.smart_randomize(rng, net10[i]) for i in range(B)]))
        pop10 = nxt[:pop]
        if deadline is not None:
            left = deadline - time.time()
            need = (cand_dt or 0.0) * len(pop10)
            print(f"  [бюджет] осталось {left:.0f} c "
                  f"(следующее поколение ~{need:.0f} c)", flush=True)
            if gen + 1 < gens and left < need + reserve:
                print("  [бюджет] следующее поколение не успеваем — стоп",
                      flush=True)
                break

    # ---------- финальная доводка лучшего ----------
    g, b = split(best10)
    if deadline is not None and deadline - time.time() < reserve:
        print("  [бюджет] финальная доводка (fine) пропущена", flush=True)
    else:
        g, b, _ = run_restart(qaoa, ht, g, b, 0, lr, fine_steps, fine_lr,
                              "доводка (fine)")

    # ---------- L-BFGS-доводка («final refinement» из MCTech) ----------
    if refine:
        g_np, b_np = g.cpu().numpy(), b.cpu().numpy()

        def p_one(i, gv, bv):
            with torch.no_grad():
                return qaoa.p_ground(
                    ht[i:i + 1],
                    torch.as_tensor(gv, dtype=torch.float32,
                                    device=dev).view(1, P),
                    torch.as_tensor(bv, dtype=torch.float32,
                                    device=dev).view(1, P)).item()

        t0 = time.time()
        n_ref = n_better = n_worse = 0
        for i in range(B):
            if deadline is not None and i > 0:
                left = deadline - time.time()
                per = (time.time() - t0) / i
                if left < max(per, 0.0) + reserve:
                    print(f"  [бюджет] L-BFGS-доводка остановлена на инстансе "
                          f"{i}/{B} (~{per * (B - i):.0f} c не хватило)",
                          flush=True)
                    break
            p_old = p_one(i, g_np[i], b_np[i])
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
            g_new = gi.detach().cpu().numpy()
            b_new = bi.detach().cpu().numpy()
            p_new = p_one(i, g_new, b_new)
            # L-BFGS с lr=1.0 без line search может перепрыгнуть оптимум:
            # принимаем результат ТОЛЬКО если он не хуже (keep best).
            if p_new > p_old:
                g_np[i], b_np[i] = g_new, b_new
                n_better += 1
            elif p_new < p_old - 1e-6:
                n_worse += 1
            n_ref = i + 1
        g, b = (torch.tensor(g_np, device=dev), torch.tensor(b_np, device=dev))
        print(f"  L-BFGS-доводка: {time.time() - t0:.0f} c "
              f"({n_ref}/{B} инстансов; улучшилось {n_better}, "
              f"отклонено как худшие {n_worse})", flush=True)
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
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="жёсткий бюджет секунд на весь инференс (0 = без "
                         "ограничения). Для h_test ставьте 520: полировка не "
                         "начинает этап, который не успевает, и submission.csv "
                         "записывается всегда")
    args = ap.parse_args()
    t_start = time.time()
    deadline = (t_start + args.time_budget) if args.time_budget > 0 else None
    reserve = budget_reserve(args.time_budget)
    if deadline is not None:
        print(f"бюджет времени: {args.time_budget:.0f} c "
              f"(резерв {reserve:.0f} c на хвост: оценка + запись CSV)",
              flush=True)

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
    n_total = len(h)
    if args.limit > 0:
        h = h[:args.limit]
        print(f"(ТЕСТОВЫЙ РЕЖИМ: только первые {len(h)} из {n_total})")

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
                            seed=args.seed, deadline=deadline,
                            reserve=reserve)
    else:
        g, b = polish_strong(qaoa, ht, g0, b0, R, S, FS, POLISH_FINE_LR,
                             POLISH_LR, seed=args.seed, deadline=deadline,
                             reserve=reserve)
    dt = time.time() - t_total

    with torch.no_grad():
        p_after = qaoa.p_ground(ht, g, b).mean().item()
    print(f"\nP(ground) после полировки: {p_after:.4f}  (сеть давала {p_before:.4f})")
    print(f"время полировки: {dt:.1f} c (лимит 600 c)")
    dt_all = time.time() - t_start
    print(f"время всего инференса: {dt_all:.1f} c"
          + (f" / бюджет {args.time_budget:.0f} c" if deadline else ""))
    if deadline is not None and dt_all > args.time_budget:
        print("ВНИМАНИЕ: бюджет превышен — посылка всё равно будет записана")

    g_np, b_np = g.cpu().numpy(), b.cpu().numpy()
    if len(np.unique(g_np, axis=0)) <= 1:
        print("ВНИМАНИЕ: все углы одинаковы — посылка вырождена "
              "(проверьте model.pt / профиль)", flush=True)
    cols = (["id"] + [f"gamma_{k}" for k in range(5)]
            + [f"beta_{k}" for k in range(5)])
    out = np.concatenate(
        [np.arange(len(h))[:, None], g_np, b_np], axis=1)
    np.savetxt(args.out, out, delimiter=",",
               header=",".join(cols), comments="", fmt=["%d"] + ["%.12f"] * 10)
    print(f"сохранено: {args.out}")

    if brain is not None:
        with torch.no_grad():
            p_final = qaoa.p_ground(ht, g, b)
        brain.update_from_results(
            [(h[i], np.concatenate([g_np[i], b_np[i]]), p_final[i].item())
             for i in range(len(h))])
        brain.save(os.path.join(DATA, "brain.pkl"))
        print(f"мозг обновлён: {len(brain)} схем -> "
              f"{os.path.join(DATA, 'brain.pkl')}")


if __name__ == "__main__":
    main()
