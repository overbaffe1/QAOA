"""Быстрая оценка P(ground) для набора углов по файлу h.

Два входа:
  --angles data/labels.npz   npz с ключами gamma, beta (N,5)
  --csv submission.csv       готовая посылка (id, gamma_0..4, beta_0..4)

Сверка с лидербордом (публичный лидерборд = сырое среднее P(ground)
по строкам CSV, без нормализации):
    python evaluate.py --angles data/labels.npz --h h_train.npy
    python evaluate.py --csv ../submission.csv --h h_train.npy
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROOT  # noqa: E402
from QAOA import QAOA  # noqa: E402


def angles_from_csv(path):
    """(gamma, beta) из submission.csv; значения читаются float() — как на
    платформе, поэтому любая нечисловая ячейка падает здесь, а не там."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        rows = [[float(c) for c in ln.strip().split(",")]
                for ln in f if ln.strip()]
    a = np.array(rows, dtype=np.float64)
    if header and header[0].strip().lower() != "id":
        raise ValueError(f"неожиданный заголовок: {header}")
    exp = ["id"] + [f"gamma_{k}" for k in range(5)] \
        + [f"beta_{k}" for k in range(5)]
    if header != exp:
        print(f"ВНИМАНИЕ: заголовок {header} != ожидаемый {exp}")
    ids = a[:, 0]
    if not np.array_equal(ids, np.arange(len(ids))):
        print("ВНИМАНИЕ: id не 0..N-1 по порядку")
    return a[:, 1:6], a[:, 6:11]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", default=None,
                    help="npz с gamma, beta (N,5)")
    ap.add_argument("--csv", default=None,
                    help="submission.csv (id, gamma_0..4, beta_0..4)")
    ap.add_argument("--h", default="h_train.npy")
    args = ap.parse_args()
    if bool(args.angles) == bool(args.csv):
        sys.exit("нужно ровно одно из --angles / --csv")

    h_path = args.h if os.path.exists(args.h) else os.path.join(ROOT, args.h)
    h = np.load(h_path)
    if args.csv:
        g_np, b_np = angles_from_csv(args.csv)
        print(f"CSV: {args.csv}")
    else:
        a = np.load(args.angles)
        g_np, b_np = a["gamma"], a["beta"]
    if len(g_np) != len(h):
        sys.exit(f"строк углов {len(g_np)} != инстансов {len(h)}")
    J = np.load(os.path.join(ROOT, "J.npy"))
    qaoa = QAOA(J)

    g = torch.tensor(np.asarray(g_np), dtype=torch.float32)
    b = torch.tensor(np.asarray(b_np), dtype=torch.float32)
    ht = torch.tensor(h, dtype=torch.float32)
    with torch.no_grad():
        p = qaoa.p_ground(ht, g, b).numpy()
    print(f"N = {len(h)}")
    print(f"P(ground): mean = {p.mean():.5f}, median = {np.median(p):.5f}, "
          f"min = {p.min():.5f}, max = {p.max():.5f}")
    print(f"сырое среднее (то, что видит лидерборд) = "
          f"{np.asarray(p, dtype=np.float64).mean():.16f}")


if __name__ == "__main__":
    main()
