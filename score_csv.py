"""score_csv.py — точный локальный скоринг CSV углов (id,gamma_0..4,beta_0..4).

  python score_csv.py lotto_ratchet.csv                 # mean, >0.5, top-15, файл <name>_p.csv
  python score_csv.py lotto_ratchet.csv merged2.csv     # + сравнение: сколько инстансов лучше ref

Точность = та же FastQAOA, что и в лотерее/симуляторе.
"""
import sys

import numpy as np
import torch

from QAOA import FastQAOA, P


def load_angles(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    A = np.stack([np.asarray(d[c], dtype=np.float32)
                  for c in d.dtype.names if c != "id"], axis=1)
    return A


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = FastQAOA(np.load("J.npy"), device=dev)
    h = torch.tensor(np.load("h_train.npy").astype(np.float32), device=dev)
    A = load_angles(sys.argv[1])
    H = A.shape[0]
    pv = q.p_ground(h, torch.tensor(A[:, :P]), torch.tensor(A[:, P:])).cpu().numpy()
    print(f"{sys.argv[1]}: mean={pv.mean():.6f}  >0.5: {(pv > 0.5).sum()}/{H}  max={pv.max():.5f}")
    order = np.argsort(-pv)
    print("top-15 (эти полировать):")
    for i in order[:15]:
        print(f"  #{i}: P={pv[i]:.6f}")
    base = sys.argv[1].rsplit(".", 1)[0]
    np.savetxt(base + "_p.csv",
               np.concatenate([np.arange(H)[:, None], pv[:, None]], axis=1),
               delimiter=",", header="id,P", comments="")
    print(f"-> {base}_p.csv (id,P для всех {H})")
    if len(sys.argv) > 2:
        B = load_angles(sys.argv[2])
        pv2 = q.p_ground(h, torch.tensor(B[:, :P]), torch.tensor(B[:, P:])).cpu().numpy()
        print(f"{sys.argv[2]}: mean={pv2.mean():.6f}  >0.5: {(pv2 > 0.5).sum()}/{H}")
        wins = pv > pv2 + 1e-12
        merged = np.where(wins, pv, pv2)
        d = pv - pv2
        print(f"{sys.argv[1]} лучше {sys.argv[2]} на {int(wins.sum())} из {H} инстансов")
        print(f"mean(max(ratchet,ref)) = {merged.mean():.6f}  (Δ к ref: {merged.mean() - pv2.mean():+.6f})")
        print("top-10 по выигрышу (ratchet - ref):")
        for i in np.argsort(-d)[:10]:
            print(f"  #{i}: ratchet={pv[i]:.6f}  ref={pv2[i]:.6f}  Δ={d[i]:+.6f}")


if __name__ == "__main__":
    main()