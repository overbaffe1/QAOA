"""Быстрая оценка P(ground) для набора углов по файлу h.

Аргументы углов: npz-файл (labels.npz / предсказания) с ключами gamma, beta.
Полезно для локальной сверки с лидербордом:
    python evaluate.py --angles data/labels.npz --h h_train.npy
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", required=True, help="npz с gamma, beta (N,5)")
    ap.add_argument("--h", default="h_train.npy")
    args = ap.parse_args()

    h_path = args.h if os.path.exists(args.h) else os.path.join(ROOT, args.h)
    h = np.load(h_path)
    a = np.load(args.angles)
    J = np.load(os.path.join(ROOT, "J.npy"))
    qaoa = QAOA(J)

    g = torch.tensor(a["gamma"], dtype=torch.float32)
    b = torch.tensor(a["beta"], dtype=torch.float32)
    ht = torch.tensor(h, dtype=torch.float32)
    with torch.no_grad():
        p = qaoa.p_ground(ht, g, b).numpy()
    print(f"N = {len(h)}")
    print(f"P(ground): mean = {p.mean():.5f}, median = {np.median(p):.5f}, "
          f"min = {p.min():.5f}, max = {p.max():.5f}")


if __name__ == "__main__":
    main()
