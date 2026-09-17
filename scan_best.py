"""scan_best.py — рейтинг ВСЕХ CSV с углами в папке (рекурсивно).

  python scan_best.py                     # текущая папка + все подпапки
  python scan_best.py D:\\path\\to\\QAOA   # конкретная папка

"Файл углов" = CSV с 500 строками (id 0..499) и 10 колонками
gamma_0..gamma_4, beta_0..beta_4 (порядок в файле не важен).
Всё остальное (ledger'ы, scan'ы, report'ы) пропускается автоматически.
По каждому файлу считает ТОЧНОЕ среднее P(ground) по h_train (FastQAOA).

Вывод: сортированный рейтинг (mean, >0.5, max) + ЛУЧШИЙ файл + его top-3.
"""
import csv
import hashlib
import os
import sys
import time

import numpy as np
import torch

from QAOA import FastQAOA, P

ANG_COLS = [f"gamma_{k}" for k in range(P)] + [f"beta_{k}" for k in range(P)]


def norm_name(s):
    return s.strip().lower().replace(" ", "_")


def try_load_angles(path):
    """Если файл — углы 500x10, вернуть A (500,10) float32; иначе None."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rdr = csv.reader(f)
            try:
                header = [norm_name(h) for h in next(rdr)]
            except StopIteration:
                return None
            if len(header) < 11:
                return None
            pos = []
            for name in ANG_COLS:  # все 10 обязательны
                if name not in header:
                    return None
                pos.append(header.index(name))
            idc = header.index("id") if "id" in header else 0
            A = np.zeros((500, 2 * P), dtype=np.float32)
            seen = set()
            for row in rdr:
                if len(row) <= max(pos + [idc]):
                    continue
                try:
                    i = int(float(row[idc]))
                except ValueError:
                    continue
                if i < 0 or i >= 500 or i in seen:
                    continue
                seen.add(i)
                A[i] = [float(row[j]) for j in pos]
            if len(seen) < 500:
                return None  # не полная submission
            return A
    except Exception:
        return None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = FastQAOA(np.load("J.npy"), device=dev)
    h = torch.tensor(np.load("h_train.npy").astype(np.float32), device=dev)

    paths = []
    for dirpath, _dirs, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.lower().endswith(".csv"):
                paths.append(os.path.join(dirpath, fn))
    print(f"найдено CSV: {len(paths)} | device: {dev}\n")

    results = []
    hashes = {}
    for p in paths:
        A = try_load_angles(p)
        if A is None:
            continue
        hh = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
        dup = next((n for n, h2 in hashes.items() if h2 == hh), None)
        if dup:
            print(f"  [дубль {dup}] {p}")
            continue
        hashes[p] = hh
        t0 = time.time()
        pv = q.p_ground(h, torch.tensor(A[:, :P]), torch.tensor(A[:, P:])).cpu().numpy()
        top3 = np.argsort(-pv)[:3]
        results.append({"p": p, "mean": float(pv.mean()), "gt": int((pv > 0.5).sum()),
                        "max": float(pv.max()), "top3": [(int(i), float(pv[i])) for i in top3],
                        "dt": time.time() - t0})

    if not results:
        print("ни один CSV не похож на файл углов 500x10")
        return

    results.sort(key=lambda r: -r["mean"])
    print(f"{'mean':>10} {'>0.5':>6} {'max':>9}  файл")
    for r in results:
        mark = "   <== BEST" if r is results[0] else ""
        print(f"{r['mean']:10.6f} {r['gt']:6d} {r['max']:9.5f}  {r['p']}{mark}")
    best = results[0]
    print(f"\nЛУЧШИЙ: {best['p']}")
    print(f"  mean={best['mean']:.6f}   >0.5: {best['gt']}/500   max={best['max']:.5f}")
    print("  top-3: " + ", ".join(f"#{i}={v:.4f}" for i, v in best["top3"]))


if __name__ == "__main__":
    main()
