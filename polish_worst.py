"""polish_worst.py — «часовой» улучшатель лучшего CSV. Детерминированный, НЕ лотерея.

  python polish_worst.py                                  # ref=merged2.csv, худшие 350, 55 мин
  python polish_worst.py --ref submission_knn2c.csv --worst 300 --minutes 55

Почему это не лотерея: лотерея бросала 4.1 млн случайных наборов и не нашла
ни одной точки лучше merged2 (0/500). Но случайный бросок — не поиск пика.
Здесь: от ТЕКУЩЕГО угла каждого худшего инстанса идём по наклонной вверх
(координатный спуск): по одному углу за раз, сетка вокруг текущего значения,
берём лучшее. Каждый шаг — ratchet: значение никогда не ухудшается.

Логика:
  1. Углы из --ref, P по h_train.
  2. Худшие --worst инстансов (именно они тянут среднее вниз).
  3. Для каждого из 10 углов: сетка 33 точек (±размах, шаг = размах/16),
     батчами на GPU. Точка лучше текущей -> берём её.
  4. Проходы с сужением размаха: 0.5 -> 0.25 -> 0.125 -> 0.0625 -> 0.03125
     -> 0.015625 (и держимся на последнем до конца времени).
  5. polish_ratchet.csv = для ВСЕХ 500 лучшее из (ref, polish) — готов к загрузке.

Время: ~1-3 сек на (угол x батч). 55 минут = десятки проходов — хватает,
чтобы дожать все худшие инстансы до локальных пиков.
"""
import argparse
import time

import numpy as np
import torch

from QAOA import FastQAOA, P

CHUNK = [8000]  # траекторий за GPU-вызов (автосократится при OOM)


def p_all(q, h, A):
    return q.p_ground(h, torch.tensor(A[:, :P]), torch.tensor(A[:, P:])).cpu().numpy()


def p_batch(q, h_rep, g, b):
    try:
        return q.p_ground(h_rep, torch.tensor(g), torch.tensor(b)).cpu().numpy()
    except RuntimeError as e:
        if "memory" in str(e).lower():
            CHUNK[0] = max(512, CHUNK[0] // 2)
            torch.cuda.empty_cache()
            print(f"  !! OOM — новый batch {CHUNK[0]}", flush=True)
            return p_batch(q, h_rep, g, b)
        raise


def sweep_angle(q, h, A, Pv, sel, k, deltas):
    """Сетка по одному углу k для инстансов sel (батчами). A, Pv обновляются."""
    n = len(sel)
    g_per_call = max(1, CHUNK[0] // len(deltas))
    for c0 in range(0, n, g_per_call):
        sel_c = sel[c0:c0 + g_per_call]
        m = len(sel_c)
        h_rep = h[sel_c].repeat_interleave(len(deltas), axis=0)
        g = np.repeat(A[sel_c][:, :P], len(deltas), axis=0).astype(np.float32)
        b = np.repeat(A[sel_c][:, P:], len(deltas), axis=0).astype(np.float32)
        for j, dv in enumerate(deltas):
            sl = slice(j * m, (j + 1) * m)
            if k < P:
                g[sl, k] += dv
            else:
                b[sl, k - P] += dv
        pv = p_batch(q, h_rep, g, b).reshape(m, len(deltas))
        bi = pv.argmax(axis=1)
        for ii, si in enumerate(sel_c):
            if pv[ii, bi[ii]] > Pv[si]:
                Pv[si] = pv[ii, bi[ii]]
                if k < P:
                    A[si, k] += deltas[bi[ii]]
                else:
                    A[si, k - P] += deltas[bi[ii]]
    return Pv


def save(A, out):
    cols = ["id"] + [f"gamma_{k}" for k in range(P)] + [f"beta_{k}" for k in range(P)]
    np.savetxt(out, np.concatenate([np.arange(A.shape[0])[:, None], A], axis=1),
               delimiter=",", header=",".join(cols), comments="",
               fmt=["%d"] + ["%.12f"] * (2 * P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="merged2.csv")
    ap.add_argument("--worst", type=int, default=350,
                    help="сколько худших инстансов полировать")
    ap.add_argument("--minutes", type=float, default=55.0)
    ap.add_argument("--out", default="polish_ratchet.csv")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = FastQAOA(np.load("J.npy"), device=dev)
    h = torch.tensor(np.load("h_train.npy").astype(np.float32), device=dev)
    print(f"device: {dev} | chunk {CHUNK[0]}", flush=True)

    d = np.genfromtxt(args.ref, delimiter=",", names=True)
    A = np.stack([np.asarray(d[c], dtype=np.float32)
                  for c in d.dtype.names if c != "id"], axis=1)
    H = A.shape[0]
    Pv = p_all(q, h, A)
    base = Pv.mean()
    gt0 = int((Pv > 0.5).sum())
    print(f"ref {args.ref}: mean={base:.6f}  >0.5: {gt0}/{H}  max={Pv.max():.5f}")

    sel = np.argsort(Pv)[:args.worst]
    Pv0 = Pv[sel].copy()
    print(f"полируем худшие {len(sel)}: P от {Pv[sel[0]]:.6f} до {Pv[sel[-1]]:.6f}")
    save(A, args.out)  # чекпоинт: на случай Ctrl+C в самом начале

    t_end = time.time() + args.minutes * 60
    phases = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
    phase = 0
    pass_no = 0
    while time.time() < t_end:
        pass_no += 1
        span = phases[min(phase, len(phases) - 1)]
        deltas = np.linspace(-span, span, 33, dtype=np.float32)
        for k in range(2 * P):
            if time.time() > t_end:
                break
            t0 = time.time()
            sweep_angle(q, h, A, Pv, sel, k, deltas)
            print(f"  pass {pass_no} (±{span:g}) уг.{k}: {time.time()-t0:5.1f}s | "
                  f"mean={Pv.mean():.6f}  gain={Pv.mean()-base:+.6f}  "
                  f"left={int(t_end-time.time())}s", flush=True)
        save(A, args.out)
        if phase < len(phases) - 1 and time.time() < t_end - 300:
            phase += 1  # сузился после полного прохода (если осталось >5 мин)

    save(A, args.out)
    imp = int((Pv[sel] > Pv0 + 1e-12).sum())
    print(f"\nИТОГО: улучшено {imp}/{len(sel)} худших инстансов")
    print(f"  mean: {base:.6f} -> {Pv.mean():.6f}  ({Pv.mean()-base:+.6f})")
    print(f"  >0.5: {gt0} -> {int((Pv > 0.5).sum())}/500")
    print(f"  -> {args.out}  (готов: python score_csv.py {args.out} {args.ref})")


if __name__ == "__main__":
    main()
