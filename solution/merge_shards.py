"""MERGE — слияние шардов (Colab) и любых CSV в один лучший.

Per-instance RATCHET: на каждом инстансе остаются углы с БОЛЬШИМ
P(ground). Показывает вклад каждого файла: кто сколько инстансов
улучшил и какие «поимки» дал (видимость).

  python merge_shards.py --h h_train.npy --base lucky.csv \
      shard_123.csv shard_456.csv shard_789.csv --out merged.csv

Формат CSV — обычный (id, gamma_0..4, beta_0..4).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from QAOA import QAOA, P  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_csv(path):
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    ids = d[:, 0].astype(int)
    order = np.argsort(ids)
    return ids[order], d[order, 1:11].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="h_train.npy")
    ap.add_argument("--base", required=True,
                    help="текущий лучший CSV (lucky.csv / mega.csv / ...)")
    ap.add_argument("shards", nargs="+", help="CSV для слияния (shard_*.csv)")
    ap.add_argument("--out", default="merged.csv")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h_path = args.h if os.path.exists(args.h) else os.path.join(root, args.h)
    h = np.load(h_path).astype(np.float32)
    qaoa = QAOA(np.load(os.path.join(root, "J.npy")), device=DEVICE)
    ht = torch.tensor(h, dtype=torch.float32, device=DEVICE)

    def p_eval(a10, idx):
        g = torch.tensor(a10[:, :P], dtype=torch.float32, device=DEVICE)
        b = torch.tensor(a10[:, P:], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return qaoa.p_ground(ht[idx], g, b).cpu().numpy()

    B = h.shape[0]
    ids, a_best = load_csv(args.base)
    p_best = p_eval(a_best, np.arange(B))
    p_base_mean = p_best.mean()
    print(f"база: {os.path.basename(args.base)} | mean = {p_base_mean:.4f}",
          flush=True)

    total_impr = 0
    for shard in args.shards:
        sids, a = load_csv(shard)
        m = (sids >= 0) & (sids < B)   # частичные шарды поддерживаются
        sids, a = sids[m], a[m]
        p = p_eval(a, sids)
        upd = p > p_best[sids] + 1e-12
        n = int(upd.sum())
        total_impr += n
        print(f"\n{os.path.basename(shard)}: улучшено {n}/{len(sids)}",
              flush=True)
        if n:
            gain = np.where(upd, p - p_best[sids], 0.0)
            top = np.argsort(-gain)[:8]
            for j in top:
                if gain[j] > 1e-6:
                    print(f"  POIMKA #{int(sids[j])}: {p_best[sids[j]]:.3f} "
                          f"-> {p[j]:.3f} (+{gain[j]:.3f})", flush=True)
        a_best[sids[upd]] = a[upd]
        p_best[sids] = np.maximum(p_best[sids], p)
        print(f"  текущий mean = {p_best.mean():.4f}", flush=True)

    out = np.concatenate([ids[:, None], a_best], axis=1)
    cols = (["id"] + [f"gamma_{k}" for k in range(P)]
            + [f"beta_{k}" for k in range(P)])
    np.savetxt(args.out, out, delimiter=",", header=",".join(cols),
               comments="", fmt=["%d"] + ["%.12f"] * 10)
    tail = np.argsort(p_best)[:10]
    print(f"\nИТОГ: mean = {p_best.mean():.4f} "
          f"(база была {p_base_mean:.4f}, +{p_best.mean() - p_base_mean:.4f})",
          flush=True)
    print("  tail: " + " ".join(f"#{i}:{p_best[i]:.3f}" for i in tail),
          flush=True)
    print(f"файл: {args.out} (ratchet: никогда хуже {args.base})", flush=True)


if __name__ == "__main__":
    main()
