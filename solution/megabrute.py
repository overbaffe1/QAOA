"""МЕГА-ДВИЖОК v2.3: тяжёлый оффлайн-брутфорс углов QAOA, ВСЕ методы мира.

Каждый раунд (батч = все инстансы, где возможно; поинстансные методы —
в python-цикле по инстансам, всё на GPU):

  1) 10 «умных» семян × 10 РАЗНЫХ оптимизаторов (v2.3: слабые jDE,
       SPSA, «универсальные» и SA-Cauchy вырезаны — вклад <0.06 mean
       P(ground) при 5-9 мин на метод, по таблице вклада 2 раундов
       реального прогона):
       сеть(дистиллят)   -> Adam
       сеть -> Adam       -> L-BFGS (поинстансно)
       память мозга      -> двухэтапный «энергия -> P(ground)»
       kNN-трансфер      -> Harmony Search
       preferences       -> Nelder-Mead (из оригинального VQE 2014)
       топ-схемы         -> Adam
       топ-схемы -> Adam -> L-BFGS
       3 свежих случайных -> CMA-ES / Powell / COBYLA
  2) эволюция: DE/rand/1/bin + экспоненциальное скрещивание (arXiv 2303.12186),
     2-5 поколений, элитизм, подкормка свежими случайными;
  3) глобальная заточка: L-BFGS по СРЕДНЕМУ по всем 500 инстансам сразу;
  4) kick-and-quench: удары sigma 0.3..1.2, per-instance ratchet;
  5) финальная заточка: L-BFGS + Adam мелким lr.

Комбинация «умная» (как ты и просил): per-instance best-of по ВСЕМ методам
раунда и ВСЕЙ истории (brain.pkl) — худшие результаты просто не попадают
в CSV. В конце раунда печатается ТАБЛИЦА ВКЛАДА КАЖДОГО МЕТОДА.

Персистентность: brain.pkl (ratchet — только улучшения), МОНОТОННЫЙ CSV
«best of all rounds» (никогда не деградирует), дистилляция мозг->сеть
каждые N раундов. Прерывание/перезапуск безопасны.

Основа (литература):
  - CMA-ES — самый надёжный оптимизатор VQA (<3.5e3 FE на 9 кубитах,
    arXiv 2511.08289; 2506.01715: CMA-ES и iL-SHADE — лучшие);
  - SPSA — 2-й по надёжности под noise (2511.08289), стандарт QAOA
    (OpenQAOA, CUDA-Q);
  - SA-Cauchy, Harmony Search, SOS — robust-эвристики (2506.01715);
  - DE + локальный полир: 100% выход в ground state (2303.12186);
  - Powell/COBYLA/NM — derivative-free из QAOA meta-learning (PLoS ONE 2021);
  - concentration углов (Brandao 2018, Q-2022-07-07-759);
  - 1-слойный QAOA ~ Boltzmann (Front. QST 2024) — двухэтапность;
  - basin-hopping / warm start для GSP (arXiv 2409.09012).

Запуск:
  python megabrute.py --h h_train.npy --out mega.csv --rounds 20 --start-seed 200 --budget 2
Бюджет (GPU, 500 h): 1 ~40 мин/раунд, 2 ~1-1.5 ч, 3 ~2-2.5 ч.
Прерывать можно в любой момент: brain.pkl / model.pt / out.csv переживают
рестарт, история не теряется.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import N_ANGLES, QAOABrain  # noqa: E402
from config import DATA, ROOT  # noqa: E402
from features import build_features  # noqa: E402
from infer import run_restart  # noqa: E402
from model import QAOAAngleNet  # noqa: E402
from QAOA import QAOA, P  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# этапы раунда и их вес в процентах (для прогресс-бара в логе)
STAGES = [
    ("семена + оптимизаторы", 30),
    ("эволюция DE", 35),
    ("глобальная L-BFGS-заточка (все 500 сразу)", 5),
    ("kick-and-quench", 15),
    ("финальная заточка", 7),
    ("сохранение: мозг + CSV", 8),
]

# budget -> параметры (масштабируются --scale для смоука)
BUDGETS = {
    1: dict(adam=400, fine=80, lbfgs=80, jde=240, nm=100, sa=300, cma=30,
            hs=60, powell=25, cobyla=80, spsa=200,
            de_gens=2, de_steps=300, kicks=2, kick_st=300),
    2: dict(adam=600, fine=100, lbfgs=100, jde=320, nm=120, sa=400, cma=40,
            hs=80, powell=35, cobyla=100, spsa=300,
            de_gens=3, de_steps=400, kicks=3, kick_st=400),
    3: dict(adam=900, fine=120, lbfgs=120, jde=480, nm=160, sa=600, cma=60,
            hs=100, powell=45, cobyla=140, spsa=450,
            de_gens=5, de_steps=500, kicks=4, kick_st=500),
}


def split10(a10):
    a = torch.tensor(a10, dtype=torch.float32, device=DEVICE)
    return a[:, :P], a[:, P:]


def join10(g, b):
    return np.concatenate([g.cpu().numpy(), b.cpu().numpy()], axis=1)


def mean_p(qaoa, ht, a10):
    g, b = split10(a10)
    with torch.no_grad():
        return qaoa.p_ground(ht, g, b).mean().item()


def _eval_one(qaoa, ht, i, x10):
    x = torch.tensor(x10, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        p = qaoa.p_ground(ht[i:i + 1], x[:P].unsqueeze(0), x[P:].unsqueeze(0))
    return -p.item()  # энергия: меньше — лучше


def _batch_eval(qaoa, ht, i, xs):
    """P(ground) батчем матрицы углов (M, 10) на инстансе i.
    Один вызов вместо M одиночных — быстрее на GPU."""
    xt = torch.tensor(xs, dtype=torch.float32, device=DEVICE)
    h1 = ht[i:i + 1].expand(len(xs), 12)
    with torch.no_grad():
        p = qaoa.p_ground(h1, xt[:, :P], xt[:, P:])
    return p.cpu().numpy()


def _tick(name, i, B, step=50):
    if i % step == 0:
        print(f"    {name}: {i}/{B}", flush=True)


# --------------------------------------------------------------------- #
# оптимизаторы                                                           #
# --------------------------------------------------------------------- #
def polish_lbfgs(qaoa, ht, g, b, iters):
    """L-BFGS поинстансно (derivative-based, batch-1)."""
    g_np, b_np = g.cpu().numpy(), b.cpu().numpy()
    B = g.shape[0]
    for i in range(B):
        _tick("L-BFGS (поинстансно)", i, B)
        gi = torch.tensor(g_np[i], dtype=torch.float32, device=DEVICE,
                          requires_grad=True)
        bi = torch.tensor(b_np[i], dtype=torch.float32, device=DEVICE,
                          requires_grad=True)

        def closure():
            opt.zero_grad()
            loss = -qaoa.p_ground(ht[i:i + 1],
                                  gi.unsqueeze(0), bi.unsqueeze(0)).mean()
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([gi, bi], lr=1.0, max_iter=iters,
                                history_size=10)
        opt.step(closure)
        g_np[i] = gi.detach().cpu().numpy()
        b_np[i] = bi.detach().cpu().numpy()
    return (torch.tensor(g_np, device=DEVICE),
            torch.tensor(b_np, device=DEVICE))


def polish_lbfgs_batch(qaoa, ht, g, b, iters):
    """L-BFGS по СРЕДНЕМУ P(ground) по всем инстансам сразу — глобальная
    заточка: учитывает, что инстансы тянут в разные стороны."""
    g = g.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True)

    def closure():
        opt.zero_grad()
        loss = -qaoa.p_ground(ht, g, b).mean()
        loss.backward()
        return loss

    opt = torch.optim.LBFGS([g, b], lr=1.0, max_iter=iters, history_size=16)
    opt.step(closure)
    return g.detach(), b.detach()


def polish_jde(qaoa, ht, seed10, fe_budget, seed):
    """Адаптивный DE (jDE): локальная популяция 12 на инстанс, F/CR
    подстраиваются по успешным потомкам (идея iL-SHADE)."""
    rng = np.random.default_rng(seed)
    B, n = seed10.shape
    out = seed10.copy()
    for i in range(B):
        _tick("jDE (поинстансно)", i, B)
        pop = seed10[i] + rng.normal(0.0, 0.3, (12, n))
        pop[0] = seed10[i]
        f = -_batch_eval(qaoa, ht, i, pop)
        fe = 12
        Fm, CRm = 0.8, 0.5
        while fe < fe_budget:
            b_ = int(np.argmin(f))
            r1, r2 = [int(j) for j in rng.choice(12, 2, replace=False)]
            Fi = float(np.clip(Fm + rng.normal(0.0, 0.1), 0.1, 1.0))
            CRi = float(np.clip(CRm + rng.normal(0.0, 0.1), 0.1, 0.9))
            mutant = pop[r1] + Fi * (pop[r2] - pop[b_])
            jrand = int(rng.integers(n))
            mask = rng.random(n) <= CRi
            mask[jrand] = True
            trial = np.where(mask, mutant, pop[b_])
            ft = _eval_one(qaoa, ht, i, trial)
            if ft < f[b_]:
                pop[b_] = trial
                f[b_] = ft
                Fm = 0.1 * Fi + 0.9 * Fm
                CRm = 0.1 * CRi + 0.9 * CRm
            fe += 1
        out[i] = pop[int(np.argmin(f))]
    return out


def polish_nm(qaoa, ht, seed10, maxiter, seed):
    """Nelder-Mead — из оригинального VQE (Peruzzo 2014)."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("  (scipy не найден — Nelder-Mead/Powell/COBYLA пропущены; "
              "pip install scipy)", flush=True)
        return seed10
    out = seed10.copy()
    B = seed10.shape[0]
    for i in range(B):
        _tick("Nelder-Mead (поинстансно)", i, B)
        res = minimize(lambda x: _eval_one(qaoa, ht, i, x), out[i],
                       method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-4,
                                "fatol": 1e-6})
        out[i] = res.x
    return out


def polish_powell(qaoa, ht, seed10, maxiter, seed):
    """Powell — детерминированный derivative-free (из QAOA meta-learning)."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return seed10
    out = seed10.copy()
    B = seed10.shape[0]
    for i in range(B):
        _tick("Powell (поинстансно)", i, B)
        res = minimize(lambda x: _eval_one(qaoa, ht, i, x), out[i],
                       method="Powell",
                       options={"maxiter": maxiter, "xatol": 1e-3})
        out[i] = res.x
    return out


def polish_cobyla(qaoa, ht, seed10, maxiter, seed):
    """COBYLA — derivative-free, стандарт сравнений в VQA."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return seed10
    out = seed10.copy()
    B = seed10.shape[0]
    for i in range(B):
        _tick("COBYLA (поинстансно)", i, B)
        res = minimize(lambda x: _eval_one(qaoa, ht, i, x), out[i],
                       method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.3})
        out[i] = res.x
    return out


def polish_cmaes(qaoa, ht, seed10, max_iter, seed):
    """CMA-ES — по Hansen, reference source (tutorial arXiv 1604.00772,
    appendix C). Поинстансно, n=10. Ключевые отличия от v2.1 (был краш
    cholesky NPD на ~40-м поколении):
      * ps/pc обновляются ДИРЕКТНЫМИ векторами: zmean = sum w_i z_i (это
        ровно D^-1 B^T (xnew-xmean)/sigma) и step = (xnew-xmean)/sigma —
        БЕЗ обратно-подстановки C^-1/2, которая при вырожденном C даёт
        1e15-значения -> overflow в exp -> NaN -> краш;
      * C-обновление в reference-форме:
        C <- (1 - c1 - cmu + c1*(1-hs)*cc*(2-cc)) C + c1 pc pc^T
             + cmu sum w_i (BDz_i)(BDz_i)^T;
      * только ПОЛОЖИТЕЛЬНЫЕ веса рекомбинации (в v2.1 нормировались и
        отрицательные — последний "родитель" тянул среднее назад);
      * B, D пересобираются каждым поколением через eigh с клампом
        собственных значений [s*1e-10, s] (n=10 — дёшево), cholesky-краш
        исключён;
      * sigma-адаптация: экспонента и sigma сами зажаты (anti-overflow)."""
    rng = np.random.default_rng(seed)
    B, n = seed10.shape
    out = seed10.copy()
    lam = int(round(4.0 + 3.0 * np.log(n)))
    mu0 = lam // 2
    w_all = np.log(mu0 / 2.0 + 0.5) - np.log(np.arange(1, mu0 + 1))
    pos = w_all > 0
    w = w_all[pos] / w_all[pos].sum()
    mu = int(pos.sum())
    mueff = float((w.sum()) ** 2 / (w ** 2).sum())
    cc = (mueff + 2.0) / (n + mueff / 2.0)
    cs = (mueff + 1.0) / (n + 2.0 + mueff)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff),
              1.0 - c1)
    damps = (1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0)
             + cs)
    # E||N(0, I_n)|| — точное
    from math import gamma as _gamma
    chiN = float(np.sqrt(2.0) * _gamma((n + 1) / 2.0) / _gamma(n / 2.0))
    for i in range(B):
        _tick("CMA-ES (поинстансно)", i, B)
        xmean = seed10[i].copy().astype(float)
        sigma = 0.5
        Bm = np.eye(n)           # собственные векторы C (столбцы)
        Dm = np.ones(n)          # sqrt собственных значений C
        ps = np.zeros(n)         # sigma-путь (в B-пространстве)
        pc = np.zeros(n)         # C-путь (исходное пространство / sigma)
        C = np.eye(n)
        best, best_f = xmean.copy(), _eval_one(qaoa, ht, i, xmean)
        no_impr = 0
        for gen in range(max_iter):
            zs = rng.standard_normal((lam, n))
            xs = xmean + sigma * (zs @ (Bm * Dm).T)
            fs = -_batch_eval(qaoa, ht, i, xs)   # 1 батч вместо lam вызовов
            if not np.all(np.isfinite(fs)):
                break
            order = np.argsort(fs)
            if fs[order[0]] < best_f:
                best_f, best = fs[order[0]], xs[order[0]].copy()
                no_impr = 0
            else:
                no_impr += 1
            sel = order[:mu]
            xnew = (w[:, None] * xs[sel]).sum(axis=0)
            zmean = (w[:, None] * zs[sel]).sum(axis=0)
            # --- sigma-путь (ДИРЕКТНО: B @ zmean, без обратных) ---
            ps = ((1.0 - cs) * ps
                  + np.sqrt(cs * (2.0 - cs) * mueff) * (Bm @ zmean))
            age = gen + 1
            hstat = 1.0 if (np.linalg.norm(ps)
                            / np.sqrt(1.0 - (1.0 - cs) ** (2 * age))
                            < (1.4 + 2.0 / (n + 1))
                            * np.sqrt(1.0 - (1.0 + cc) ** (-2 * age))
                            * (1.0 + cs)) else 0.0
            # --- C-путь (исходное пространство / sigma) ---
            step = (xnew - xmean) / sigma
            pc = ((1.0 - cc) * pc
                  + hstat * np.sqrt(cc * (2.0 - cc) * mueff) * step)
            # --- C-обновление (reference-форма) ---
            BDz = (xs[sel] - xmean) / sigma
            Zwt = (BDz.T * w) @ BDz
            C = ((1.0 - c1 - cmu + c1 * (1.0 - hstat) * cc * (2.0 - cc)) * C
                 + c1 * np.outer(pc, pc)
                 + cmu * Zwt)
            C = (C + C.T) / 2.0
            # --- sigma (зажатые: экспонента и само значение) ---
            sigma *= float(np.exp(np.clip(
                cs / damps * (np.linalg.norm(ps) / chiN - 1.0),
                -10.0, 10.0)))
            sigma = float(np.clip(sigma, 1e-8, 1e6))
            xmean = xnew
            # --- робастная факторизация C (n=10: eigh дёшево) ---
            try:
                ev, Bm2 = np.linalg.eigh(C)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(ev)):
                break
            s = max(float(ev.max()), 1e-12)
            ev = np.clip(ev, s * 1e-10, s)
            Bm, Dm = Bm2, np.sqrt(ev)
            if no_impr > 2 * lam:
                break
        out[i] = best
    return out

def polish_spsa(qaoa, ht, seed10, iters, seed):
    """SPSA (Spall 1992), 2-оценочная версия: g ≈ (f(x+cd)−f(x−cd))/(2c)·d,
    обе точки за один батч-вызов. Градиент-free, стандарт QAOA."""
    B = seed10.shape[0]
    x = torch.tensor(seed10, dtype=torch.float32, device=DEVICE)
    c0, A, gamma, alpha = 0.25, 3.0, 0.6, 0.6
    for k in range(iters):
        d = (torch.randint(0, 2, (B, N_ANGLES), device=DEVICE) * 2 - 1)
        d = d.float()
        c = c0 / (k + 1) ** gamma
        a = A / (k + 1 + A) ** alpha
        xp = x + c * d
        xm = x - c * d
        with torch.no_grad():
            pp = -qaoa.p_ground(ht, xp[:, :P], xp[:, P:])
            pm = -qaoa.p_ground(ht, xm[:, :P], xm[:, P:])
        x = x - a * ((pp - pm) / (2 * c)).unsqueeze(1) * d
    return x.cpu().numpy()


def polish_sa(qaoa, ht, seed10, iters, seed):
    """Симулированное отжигание (гауссовы предложения, весь батч)."""
    rng = np.random.default_rng(seed)
    B = seed10.shape[0]
    cur = torch.tensor(seed10, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        fcur = -qaoa.p_ground(ht, cur[:, :P], cur[:, P:])
    T0, T1 = 0.30, 0.002
    for t in range(iters):
        T = T0 * (T1 / T0) ** (t / max(iters - 1, 1))
        sigma = 0.25 * (1.0 - 0.5 * t / iters)
        prop = cur + torch.randn_like(cur) * sigma
        with torch.no_grad():
            fnew = -qaoa.p_ground(ht, prop[:, :P], prop[:, P:])
        accept = (fnew < fcur) | (
            torch.rand(B, device=DEVICE).log() < (fcur - fnew) / T)
        cur = torch.where(accept[:, None], prop, cur)
        fcur = torch.where(accept, fnew, fcur)
    return cur.cpu().numpy()


def polish_sacauchy(qaoa, ht, seed10, iters, seed):
    """SA-Cauchy — отжигание с тяжёлым хвостом (robust, arXiv 2506.01715):
    Cauchy-шаги лучше перескакивают риджи."""
    rng = np.random.default_rng(seed)
    B = seed10.shape[0]
    cur = torch.tensor(seed10, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        fcur = -qaoa.p_ground(ht, cur[:, :P], cur[:, P:])
    T0, T1 = 0.30, 0.002
    for t in range(iters):
        T = T0 * (T1 / T0) ** (t / max(iters - 1, 1))
        sigma = 0.4 * (1.0 - 0.6 * t / iters)
        cauchy = torch.from_numpy(rng.standard_cauchy((B, N_ANGLES))
                                  ).to(device=DEVICE)
        prop = cur + cauchy * sigma
        with torch.no_grad():
            fnew = -qaoa.p_ground(ht, prop[:, :P], prop[:, P:])
        accept = (fnew < fcur) | (
            torch.rand(B, device=DEVICE).log() < (fcur - fnew) / T)
        cur = torch.where(accept[:, None], prop, cur)
        fcur = torch.where(accept, fnew, fcur)
    return cur.cpu().numpy()


def polish_hs(qaoa, ht, seed10, iters, seed):
    """Harmony Search (robust, arXiv 2506.01715): память гармоний,
    pitch adjustment с Cauchy-шагом."""
    rng = np.random.default_rng(seed)
    B = seed10.shape[0]
    out = seed10.copy()
    HM = 12
    n = N_ANGLES
    for i in range(B):
        _tick("Harmony Search (поинстансно)", i, B)
        mem = seed10[i] + rng.normal(0.0, 0.4, (HM, n))
        mem[0] = seed10[i]
        fmem = -_batch_eval(qaoa, ht, i, mem)
        b = int(np.argmin(fmem))
        for t in range(iters):
            par = 0.05 + 0.45 * t / max(iters - 1, 1)
            bw = 0.3 * (1.0 - 0.7 * t / max(iters - 1, 1))
            x = np.where(rng.random(n) < 0.9, mem[b],
                         rng.normal(0, 1, n))
            for j in range(n):
                if rng.random() < par:
                    x[j] += bw * rng.standard_cauchy()
            f = _eval_one(qaoa, ht, i, x)
            if f < fmem[b]:
                mem[b], fmem[b] = x, f
            else:
                w = int(np.argmax(fmem))
                if f < fmem[w]:
                    mem[w], fmem[w] = x, f
        out[i] = mem[int(np.argmin(fmem))]
    return out


def two_stage_energy(qaoa, ht, seed10, e_steps, p_steps):
    """Двухэтапность: сначала энергия (гладкий ландшафт), потом P(ground).
    Обоснование: 1-слойный QAOA ~ Boltzmann (Front. QST 2024)."""
    g, b = split10(seed10)
    g = g.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True)
    E = qaoa.energies(ht)
    opt = torch.optim.Adam([g, b], lr=0.05)
    for _ in range(e_steps):
        opt.zero_grad()
        loss = (qaoa.probs(ht, g, b) * E).sum(dim=1).mean()
        loss.backward()
        opt.step()
    opt = torch.optim.Adam([g, b], lr=0.03)
    for _ in range(p_steps):
        opt.zero_grad()
        loss = -qaoa.p_ground(ht, g, b).mean()
        loss.backward()
        opt.step()
    return join10(g.detach(), b.detach())


def learn_universal(qaoa, seed, n_h=30, steps=600):
    """Универсальные углы (concentration): один набор на типовой ансамбль
    (arXiv 1812.04170, Q-2022-07-07-759)."""
    rng = np.random.default_rng(seed)
    ht = torch.tensor(rng.uniform(-1, 1, (n_h, 12)), dtype=torch.float32,
                      device=DEVICE)
    g = (torch.rand(1, P, device=DEVICE) * 2 * np.pi).requires_grad_(True)
    b = (torch.rand(1, P, device=DEVICE) * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([g, b], lr=0.05)
    for s in range(steps):
        opt.zero_grad()
        loss = -qaoa.p_ground(ht, g.expand(n_h, P),
                              b.expand(n_h, P)).mean()
        loss.backward()
        opt.step()
        if s % 200 == 0:
            print(f"  универсальные углы {s}/{steps}: {-loss.item():.4f}",
                  flush=True)
    return np.concatenate([g.detach().cpu().numpy()[0],
                           b.detach().cpu().numpy()[0]])


def de_next_generation(t1, t2, extras, rng, pop, B, F=0.8, CR=0.5):
    """Элиты + DE/rand/1/bin + экспоненциальное скрещивание + свежие."""
    nxt = [t1, t2]
    n = N_ANGLES
    m = rng.random((B, n)) < CR
    m[:, 0] = True
    nxt.append(np.where(m, t1 + F * (t2 - extras[0]), t2))
    start = rng.integers(0, n, B)
    j = np.arange(n)
    mexp = ((j - start[:, None]) % n) < (1 + rng.integers(1, n + 1, B)[:, None])
    nxt.append(np.where(mexp, t2 + F * (t1 - extras[1]), t1))
    while len(nxt) < pop:
        nxt.append(extras[len(nxt) % len(extras)].copy())
    return nxt[:pop]


# --------------------------------------------------------------------- #
def _prog(st, frac, extra=""):
    prev = sum(w for _, w in STAGES[:st])
    pct = prev + STAGES[st][1] * min(max(frac, 0.0), 1.0)
    print(f"  [{pct:3.0f}%] {STAGES[st][0]}{extra}", flush=True)


def round_search(qaoa, ht, h, g0, b0, brain, cfg, seed, rnd, nrnd):
    rng = np.random.default_rng(seed)
    B = h.shape[0]
    g0n, b0n = g0.cpu().numpy(), b0.cpu().numpy()
    net10 = np.concatenate([g0n, b0n], axis=1)
    report = []  # (имя метода, mean P(ground))

    def adam_polish(a10, steps, fine, tag):
        g, b = split10(a10)
        gs, bs, p = run_restart(qaoa, ht, g, b, steps, 0.05,
                                fine, 0.01, tag)
        return join10(gs, bs), p.cpu().numpy()

    # ---------- 1) 10 умных семян (v2.3: слабые jDE/SPSA/универс./SA-Cauchy
    #     вырезаны — вклад <0.06 при ~5-9 мин на метод; v2.2-таблица вклада)
    # ----------
    _prog(0, 0, f"  (раунд {rnd}/{nrnd})")
    mem = np.empty((B, N_ANGLES))
    knn = np.empty((B, N_ANGLES))
    for i in range(B):
        k = brain.best_for(h[i])
        mem[i] = (k if k is not None
                  else net10[i] + rng.normal(0, 0.3, N_ANGLES))
        kk = brain.knn_transfer(h[i])
        knn[i] = kk if kk is not None else brain.smart_randomize(rng, net10[i])
    pref = np.stack([brain.smart_randomize(rng, net10[i]) for i in range(B)])
    if brain.top_schemes:
        top = brain.top_schemes[int(rng.integers(len(brain.top_schemes)))]
        top10 = top[None, :] + rng.normal(0, 0.1, (B, N_ANGLES))
    else:
        top10 = net10 + rng.normal(0, 0.2, (B, N_ANGLES))
    fresh = [np.stack([brain.smart_randomize(rng, net10[i])
                       for i in range(B)]) for _ in range(3)]

    # ---------- 2) каждый метод — свой оптимизатор ----------
    t0 = time.time()
    pool = []  # (имя, a10, mean_p)
    NM = 10

    def hdr(name, c):
        print(f"  [{(c - 1) / NM * 30:3.0f}%] -> {name} ...", flush=True)

    def add(name, a, c, pm=None):
        pm = pm if pm is not None else mean_p(qaoa, ht, a)
        pool.append((name, a, pm))
        report.append((name, pm))
        print(f"  [{c / NM * 30:3.0f}%]    {name}: {pm:.4f}", flush=True)

    c = 0
    c += 1; hdr("1 сеть+Adam", c)
    a, p = adam_polish(net10, cfg["adam"], cfg["fine"], "м1 сеть + Adam")
    add("1 сеть+Adam", a, c, p.mean())
    c += 1; hdr("2 сеть+Adam+L-BFGS", c)
    a2 = join10(*polish_lbfgs(qaoa, ht, *split10(a), cfg["lbfgs"]))
    add("2 сеть+Adam+L-BFGS", a2, c)
    c += 1; hdr("3 память: энергия->P", c)
    a = two_stage_energy(qaoa, ht, mem, 400, 300)
    add("3 память: энергия->P", a, c)
    c += 1; hdr("4 kNN+HarmonySearch", c)
    a = polish_hs(qaoa, ht, knn, cfg["hs"], seed + 3)
    add("4 kNN+HarmonySearch", a, c)
    c += 1; hdr("5 pref+NelderMead", c)
    a = polish_nm(qaoa, ht, pref, cfg["nm"], seed + 2)
    add("5 pref+NelderMead", a, c)
    c += 1; hdr("6 топ-схемы+Adam", c)
    a, p = adam_polish(top10, cfg["adam"], cfg["fine"], "м6 топ+Adam")
    add("6 топ-схемы+Adam", a, c, p.mean())
    c += 1; hdr("7 топ+Adam+L-BFGS", c)
    a2 = join10(*polish_lbfgs(qaoa, ht, *split10(a), cfg["lbfgs"]))
    add("7 топ+Adam+L-BFGS", a2, c)
    c += 1; hdr("8 fresh+CMA-ES", c)
    a = polish_cmaes(qaoa, ht, fresh[0], cfg["cma"], seed + 11)
    add("8 fresh+CMA-ES", a, c)
    c += 1; hdr("9 fresh+Powell", c)
    a = polish_powell(qaoa, ht, fresh[1], cfg["powell"], seed + 12)
    add("9 fresh+Powell", a, c)
    c += 1; hdr("10 fresh+COBYLA", c)
    a = polish_cobyla(qaoa, ht, fresh[2], cfg["cobyla"], seed + 13)
    add("10 fresh+COBYLA", a, c)
    _prog(0, 1, f": {len(pool)} методов ({time.time() - t0:.0f} c)")

    # ---------- 3) эволюция DE ----------
    pop10 = [a for _, a, _ in pool]
    extras = [fresh[0], fresh[1], fresh[2]]
    popn = len(pop10)
    best10, best_fit = None, None
    for gen in range(cfg["de_gens"]):
        fit = np.zeros((popn, B))
        opt10 = []
        for c in range(popn):
            a10, p10 = adam_polish(pop10[c], cfg["de_steps"], 0,
                                   f"эволюция gen{gen + 1} к{c + 1}")
            fit[c] = p10
            opt10.append(a10)
            _prog(1, (gen * popn + c + 1) / (cfg["de_gens"] * popn),
                  f" gen{gen + 1} к{c + 1}: {p10.mean():.4f}")
        am = fit.argmax(axis=0)
        cur10 = np.stack([opt10[am[i]][i] for i in range(B)])
        cur_fit = fit[am, np.arange(B)]
        if best10 is None:
            best10, best_fit = cur10, cur_fit
        else:
            upd = cur_fit > best_fit
            best10 = np.where(upd[:, None], cur10, best10)
            best_fit = np.maximum(best_fit, cur_fit)
        report.append((f"эволюция gen{gen + 1} best-of", best_fit.mean()))
        _prog(1, (gen + 1) / cfg["de_gens"],
              f" -> best-of mean P(ground) = {best_fit.mean():.4f}")
        if gen == 0:
            # ВИДИМОСТЬ: ген1 — это per-instance best-of по 10 методам
            # (каждый кандидат = один метод). Итог выше всех средних,
            # т.к. каждый метод побеждает на СВОЕЙ подгруппе инстансов:
            win = np.bincount(am, minlength=popn)
            parts = [f"м{c + 1}:{win[c]}"
                     + (f"(там {fit[c][am == c].mean():.3f})" if win[c] else "")
                     for c in range(popn)]
            print("     best-of поинстансно, победил: " + " | ".join(parts),
                  flush=True)
        order = np.argsort(-fit, axis=0)
        t1 = np.stack([opt10[c][i] for i, c in enumerate(order[0])])
        t2 = np.stack([opt10[c][i] for i, c in enumerate(order[1])])
        pop10 = de_next_generation(t1, t2, extras, rng, popn, B)

    # ---------- 4) глобальная L-BFGS-заточка (все 500 сразу) ----------
    _prog(2, 0.3)
    g, b = polish_lbfgs_batch(qaoa, ht, *split10(best10), 60)
    g, b, p = run_restart(qaoa, ht, g, b, 100, 0.03, 50, 0.005,
                          "глобальная заточка (среднее по 500)")
    with torch.no_grad():
        pg = qaoa.p_ground(ht, g, b).cpu().numpy()
    new10 = join10(g, b)
    upd = pg > best_fit
    best10 = np.where(upd[:, None], new10, best10)
    best_fit = np.maximum(best_fit, pg)
    report.append(("глоб. L-BFGS+Adam", best_fit.mean()))
    _prog(2, 1, f": {best_fit.mean():.4f}")

    # ---------- 5) kick-and-quench ----------
    g, b = split10(best10)
    for k in range(cfg["kicks"]):
        sig = 0.3 * (k + 1)
        gk = g + torch.randn_like(g) * sig
        bk = b + torch.randn_like(b) * sig
        gk, bk, _ = run_restart(qaoa, ht, gk, bk, cfg["kick_st"], 0.05, 50,
                                0.01, f"kick {k + 1}/{cfg['kicks']} "
                                      f"(sigma={sig:.1f})")
        with torch.no_grad():
            pk = qaoa.p_ground(ht, gk, bk)
            pc = qaoa.p_ground(ht, g, b)
        upd = (pk > pc).unsqueeze(1)
        g, b = torch.where(upd, gk, g), torch.where(upd, bk, b)
        best_fit = np.maximum(best_fit, pk.cpu().numpy())
        _prog(3, (k + 1) / cfg["kicks"], f" kick {k + 1}: {best_fit.mean():.4f}")
    report.append(("kick-and-quench", best_fit.mean()))

    # ---------- 6) финальная заточка ----------
    _prog(4, 0.2)
    g, b = polish_lbfgs(qaoa, ht, g, b, cfg["lbfgs"])
    g, b, p = run_restart(qaoa, ht, g, b, 150, 0.05, 100, 0.005,
                          "финальная заточка")
    pn = p.cpu().numpy()
    out10 = join10(g, b)
    upd = pn > best_fit
    best10 = np.where(upd[:, None], out10, best10)
    best_fit = np.maximum(best_fit, pn)
    report.append(("финальная заточка", best_fit.mean()))
    _prog(4, 1, f": {best_fit.mean():.4f}")

    # ---------- таблица вклада ----------
    print("\n  ══ ВКЛАД МЕТОДОВ (mean P(ground), этот раунд) ══", flush=True)
    for name, v in sorted(report, key=lambda t: -t[1]):
        bar = "#" * int(v * 100)
        print(f"    {name:<30s} {v:.4f} {bar}", flush=True)
    print("  ══ умная комбинация: best-of по ВСЕМ методам и истории ══",
          flush=True)
    print("    (итог = лучший вариант ПО КАЖДОМУ инстансу, поэтому он",
          flush=True)
    print("     выше любого среднего из таблицы; кто где выиграл — в строке",
          flush=True)
    print("     'best-of поинстансно' после эволюции gen1)", flush=True)
    return best10, torch.tensor(best_fit, device=DEVICE)


# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="h_train.npy")
    ap.add_argument("--out", default="mega.csv")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--start-seed", type=int, default=200)
    ap.add_argument("--budget", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--distill-every", type=int, default=2,
                    help="каждые N раундов докормить сеть мозгом (0 = нет)")
    ap.add_argument("--min-improve", type=float, default=0.0002,
                    help="стоп, если история не выросла на N за 3 раунда")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="скрытый: множитель всех бюджетов (0.1 — смоук)")
    args = ap.parse_args()
    cfg = {k: (v if isinstance(v, str) else
               max(4, int(round(v * args.scale))))
           for k, v in BUDGETS[args.budget].items()}

    h_path = args.h if os.path.exists(args.h) else os.path.join(ROOT, args.h)
    h = np.load(h_path)
    J = np.load(os.path.join(ROOT, "J.npy"))
    qaoa = QAOA(J, device=DEVICE)
    X, _ = build_features(J, h)

    def load_net():
        ckpt = torch.load(os.path.join(DATA, "model.pt"), map_location="cpu",
                          weights_only=True)
        net = QAOAAngleNet(in_dim=ckpt["in_dim"]).to(DEVICE)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net

    net = load_net()
    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    ht = torch.tensor(h, dtype=torch.float32, device=DEVICE)
    brain = QAOABrain.load(os.path.join(DATA, "brain.pkl"))
    print(f"device: {DEVICE}, N = {len(h)}, бюджет: {args.budget}, "
          f"мозг: {len(brain)} схем")


    cols = (["id"] + [f"gamma_{k}" for k in range(5)]
            + [f"beta_{k}" for k in range(5)])
    hist_best = -1.0
    stagnant = 0
    for r in range(args.rounds):
        seed = args.start_seed + r
        t0 = time.time()
        print(f"\n########## РАУНД {r + 1}/{args.rounds} (seed {seed}) "
              f"##########", flush=True)
        with torch.no_grad():
            g0, b0 = net.angles(xt)
        best10, p_round = round_search(qaoa, ht, h, g0, b0, brain, cfg,
                                       seed, r + 1, args.rounds)
        _prog(5, 0.2, ": мозг")
        brain.update_from_results(
            [(h[i], best10[i], p_round[i].item()) for i in range(len(h))])
        # МОНОТОННЫЙ CSV: для каждого инстанса — лучший из всей истории
        all10 = np.empty((len(h), N_ANGLES))
        for i in range(len(h)):
            a = brain.best_for(h[i])
            all10[i] = a if a is not None else best10[i]
        with torch.no_grad():
            p_hist = qaoa.p_ground(
                ht, torch.tensor(all10[:, :P], dtype=torch.float32,
                                 device=DEVICE),
                torch.tensor(all10[:, P:], dtype=torch.float32,
                             device=DEVICE))
        out = np.concatenate([np.arange(len(h))[:, None], all10], axis=1)
        np.savetxt(args.out, out, delimiter=",", header=",".join(cols),
                   comments="", fmt=["%d"] + ["%.12f"] * 10)
        brain.save(os.path.join(DATA, "brain.pkl"))
        _prog(5, 1.0, f": CSV ({args.out})")
        hist = p_hist.mean().item()
        print(f"РАУНД {r + 1}: этот раунд {p_round.mean().item():.4f} | "
              f"ИСТОРИЯ (best of all) {hist:.4f} | "
              f"({time.time() - t0:.0f} c)", flush=True)
        if hist <= hist_best + args.min_improve:
            stagnant += 1
        else:
            stagnant = 0
        hist_best = max(hist_best, hist)
        if stagnant >= 3 and args.rounds > 1:
            print(f"плато: история не растёт {stagnant} раунда подряд — "
                  f"стоп. Лучшее: {hist_best:.4f}", flush=True)
            break
        if args.distill_every and (r + 1) % args.distill_every == 0:
            print("дистилляция мозг -> сеть ...", flush=True)
            import distill
            distill.STEPS = 1500
            distill.main()
            net = load_net()
            with torch.no_grad():
                g0, b0 = net.angles(xt)
            print(f"сеть после дистилляции: "
                  f"{qaoa.p_ground(ht, g0, b0).mean().item():.4f}",
                  flush=True)
    print(f"\nГОТОВО. Итог (история): {hist_best:.4f}, файл: {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
