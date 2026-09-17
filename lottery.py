"""LOTTERY v4 — fast random angle search, GPU-batched, PERSISTENT state.

Speed
  * GPU-batched: 8 random CSVs per CUDA call, one sync per batch,
    angle generation on CPU pipelined with GPU eval.
    GPU: ~30-60 sets/s  ->  500 sets in ~10-20 s, 10 000 sets in ~3-6 min.
    CPU (fallback): ~0.5-0.7 sets/s with 2 threads.
  * CUDA OOM -> auto-halves --batch and retries.
  * Why no quantization: the simulator is EXACT 4096-state evolution in
    float32 (error ~1e-6). int8/half would inject >1e-3 noise, and the
    ratchet compares P exactly at that scale. The bottleneck is kernel
    throughput — the GPU batch solves that directly.
  * Why no Cython/Numba/JIT: the heavy work already runs as C++/CUDA
    kernels (FastQAOA group-mixer); Python only submits batches (ms of
    overhead per ~200 ms of GPU work). No interpreter wrapper can beat
    that.

Smart random families (measured on 500 h_train, 80 sets each):
  informed  gamma,beta ~ U(0,0.6) + 1-3 layers beta = pi/2 + N(0, 0.15):
            set-mean 0.00348 (11x uniform), best instance 0.578 (17x),
            190/500 instances >= 0.05  ->  default family.
  uniform   gamma,beta ~ U(0,2pi): 0.00031 (= ~1/4096 uniform state).
  wide      gamma ~ U(-6pi,6pi): 0.00026 — WORSE than uniform (large gamma
            winds the phase to near-uniform states).
  wideinf   wide gamma + informed beta: 0.00027 — large gamma destroys it.
  Note: beta period is pi, not 2pi (RX(beta+pi) = -RX(beta), verified to
        1e-8), so U(0,2pi) wastes half the draws.

Persistent state (survives restarts, all in <prefix>_* files):
  * <prefix>_seen.txt     — every seed ever used: RESTARTS NEVER REPEAT
  * <prefix>_state.csv    — history: one row per run
  * <prefix>_ratchet.csv  — per-instance BEST angles over ALL runs
  * ALLTIME best (set-mean, single instance) carried across runs and
    shown on every line.

CPU threads stay low by default so it can share the PC with grind
(on GPU it does not touch the CPU at all).

Usage (from the QAOA folder):
    python lottery.py --h h_train.npy --ref merged2.csv            # 500 sets
    python lottery.py --h h_train.npy --ref merged2.csv --sets 10000
    python lottery.py --h h_train.npy --ref merged2.csv --batch 16 # if VRAM allows
    python lottery.py --h h_train.npy --ref merged2.csv --mix "uniform:1.0"
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from QAOA import FastQAOA, P  # noqa: E402

FAMS = ("uniform", "wide", "informed", "wideinf")


def parse_mix(s):
    out = {}
    for part in s.split(","):
        name, _, w = part.partition(":")
        name = name.strip()
        assert name in FAMS, f"unknown family {name} (use {FAMS})"
        out[name] = float(w if w else 1.0)
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


def gen_family(fam, rng, H):
    g = np.empty((H, P)); b = np.empty((H, P))
    if fam == "uniform":
        g[:] = rng.uniform(0.0, 2 * np.pi, (H, P))
        b[:] = rng.uniform(0.0, 2 * np.pi, (H, P))
    elif fam == "wide":
        g[:] = rng.uniform(-6 * np.pi, 6 * np.pi, (H, P))
        b[:] = rng.uniform(0.0, np.pi, (H, P))
    elif fam == "informed":
        g[:] = rng.uniform(0.0, 0.6, (H, P))
        b[:] = rng.uniform(0.0, 0.6, (H, P))
        for i in range(H):
            nl = int(rng.integers(1, 4))
            for l in rng.choice(P, nl, replace=False):
                b[i, l] = np.pi / 2 + rng.normal(0.0, 0.15)
    elif fam == "wideinf":
        g[:] = rng.uniform(-6 * np.pi, 6 * np.pi, (H, P))
        b[:] = rng.uniform(0.0, 0.6, (H, P))
        for i in range(H):
            nl = int(rng.integers(1, 4))
            for l in rng.choice(P, nl, replace=False):
                b[i, l] = np.pi / 2 + rng.normal(0.0, 0.15)
    return g, b


def fam_of(rng, mix):
    r = rng.random()
    acc = 0.0
    for name, w in mix.items():
        acc += w
        if r <= acc:
            return name
    return next(iter(mix))


def load_seen(path):
    if os.path.exists(path):
        with open(path) as f:
            return set(int(ln.split()[0]) for ln in f if ln.strip())
    return set()


def load_history(path):
    rows = []
    if os.path.exists(path):
        d = np.genfromtxt(path, delimiter=",", names=True)
        if d.ndim == 0:
            rows.append({c: d[c] for c in d.dtype.names})
        else:
            for i in range(len(d)):
                rows.append({c: d[c][i] for c in d.dtype.names})
    return rows


def load_all_sets(path):
    """Постоянный леджер всех сидов: seed,family,set_mean,max_inst,best_h[,run_id]."""
    rows = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("seed"):
                    continue
                seed = int(float(r["seed"]))
                rows[seed] = (seed, r.get("family", ""),
                              float(r["set_mean"]), float(r["max_inst"]),
                              int(float(r["best_h"])),
                              (int(float(r["run_id"])) if r.get("run_id") else 0))
    return rows


def write_csv(path, header, rows, fmts):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(fm % v for fm, v in zip(fmts, r)) + "\n")


def pick_seeds(seen, base, total):
    seeds, i = [], 0
    while len(seeds) < total:
        cand = int(base) + (len(seen) + i) * 7919
        i += 1
        if cand not in seen:
            seeds.append(cand)
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="h_train.npy")
    ap.add_argument("--ref", default="merged2.csv",
                    help="CSV with current best angles (id,gamma_*,beta_*)")
    ap.add_argument("--sets", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0,
                    help="0 = random base seed each restart (default)")
    ap.add_argument("--batch", type=int, default=8,
                    help="sets per CUDA/CPU call (8=4000 trajectories; "
                         "lower if OOM)")
    ap.add_argument("--threads", type=int, default=2,
                    help="CPU threads (CPU mode only)")
    ap.add_argument("--mix", default="informed:0.7,uniform:0.3",
                    help="family weights, e.g. informed:0.7,uniform:0.3; "
                         "families: informed, uniform, wide, wideinf")
    ap.add_argument("--save-top", type=int, default=10)
    ap.add_argument("--prefix", default="lotto")
    args = ap.parse_args()

    mix = parse_mix(args.mix)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        torch.set_num_threads(max(1, args.threads))
    q = FastQAOA(np.load("J.npy"), device=DEVICE)
    h = torch.tensor(np.load(args.h).astype(np.float32), device=DEVICE)
    H = h.shape[0]
    print(f"device: {DEVICE} | batch {args.batch} | "
          f"~{args.batch} sets per eval call", flush=True)

    # ---- persistent state -------------------------------------------
    seen_p = f"{args.prefix}_seen.txt"
    hist_p = f"{args.prefix}_state.csv"
    rat_p = f"{args.prefix}_ratchet.csv"
    all_p = f"{args.prefix}_all.csv"
    top_p = f"{args.prefix}_top.csv"
    prev_all = load_all_sets(all_p)
    seen = load_seen(seen_p)
    hist = load_history(hist_p)
    run_id = len(hist) + 1

    best_a = None
    if args.ref and os.path.exists(args.ref):
        d = np.genfromtxt(args.ref, delimiter=",", names=True)
        best_a = np.stack([np.asarray(d[c], dtype=float) for c in d.dtype.names
                           if c != "id"], axis=1)
        best_p = q.p_ground(h, torch.tensor(best_a[:, :P], device=DEVICE),
                            torch.tensor(best_a[:, P:], device=DEVICE)).cpu().numpy()
        print(f"ref {args.ref}: mean P = {best_p.mean():.4f}", flush=True)
    else:
        best_a = np.zeros((H, 2 * P))
        best_p = np.zeros(H)
        print("no --ref: pure lottery", flush=True)
    if os.path.exists(rat_p):
        d = np.genfromtxt(rat_p, delimiter=",", names=True)
        ra = np.stack([np.asarray(d[c], dtype=float) for c in d.dtype.names
                       if c != "id"], axis=1)
        rp = q.p_ground(h, torch.tensor(ra[:, :P], device=DEVICE),
                        torch.tensor(ra[:, P:], device=DEVICE)).cpu().numpy()
        upd = rp > best_p
        best_a[upd] = ra[upd]
        best_p[upd] = rp[upd]
        print(f"ratchet {rat_p}: mean P = {rp.mean():.4f} "
              f"({int(upd.sum())} instances better than ref)", flush=True)

    at_mean = float(max([r["best_mean"] for r in hist], default=-1.0))
    at_mean_seed = int(max(hist, key=lambda r: r["best_mean"])["best_seed"]) \
        if hist else None
    at_mean_run = int(max(hist, key=lambda r: r["best_mean"])["run_id"]) \
        if hist else 0
    at_i = float(max([r["best_inst"] for r in hist], default=-1.0))
    at_i_seed = int(max(hist, key=lambda r: r["best_inst"])["best_inst_seed"]) \
        if hist else None
    at_i_h = int(max(hist, key=lambda r: r["best_inst"])["best_h"]) if hist else 0
    at_i_run = int(max(hist, key=lambda r: r["best_inst"])["run_id"]) if hist else 0

    base = int(args.seed) if args.seed else \
        random.SystemRandom().randint(1, 2 ** 31 - 1)
    seeds = pick_seeds(seen, base, args.sets)
    print(f"run #{run_id}: {args.sets} fresh seeds, base={base}, "
          f"already seen={len(seen)}, mix={ {k: round(v, 2) for k, v in mix.items()} }",
          flush=True)
    print(f"ALLTIME before this run: set-mean {at_mean:.5f} "
          f"(seed {at_mean_seed}, run #{at_mean_run}) | "
          f"inst {at_i:.4f} @h{at_i_h} (seed {at_i_seed}, run #{at_i_run})",
          flush=True)

    # ---- loop (batched GPU eval) --------------------------------------
    t0 = time.time()
    total = args.sets
    wins = 0
    seen_f = open(seen_p, "a")
    run_rows = []
    run_best_mean, run_best_mean_seed = -1.0, None
    run_best_i, run_best_i_seed, run_best_i_h = -1.0, None, 0
    batch = max(1, args.batch)
    n_done = 0
    first_batch_time = None

    b0 = 0
    while b0 < total:
        nb = min(batch, total - b0)
        seeds_b = seeds[b0:b0 + nb]
        A = np.empty((nb, H, 2 * P), dtype=np.float32)
        fams_b = []
        for j, s in enumerate(seeds_b):
            rng = np.random.default_rng(int(s))
            fam = fam_of(rng, mix)
            fams_b.append(fam)
            g, b = gen_family(fam, rng, H)
            A[j, :, :P] = g
            A[j, :, P:] = b
            seen_f.write(f"{int(s)} {fam}\n")
            seen_f.flush()
            seen.add(int(s))
        try:
            gt = torch.from_numpy(A[:, :, :P].reshape(-1, P)).to(
                DEVICE, non_blocking=True)
            bt = torch.from_numpy(A[:, :, P:].reshape(-1, P)).to(
                DEVICE, non_blocking=True)
            t_b = time.time()
            with torch.no_grad():
                pv = q.p_ground(h.repeat(nb, 1), gt, bt).cpu().numpy()
            dt_b = time.time() - t_b
            pv = pv.reshape(nb, H)
        except RuntimeError as e:
            if DEVICE == "cuda" and "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                batch = max(1, batch // 2)
                print(f"  !! CUDA OOM on batch {nb} — new batch {batch} "
                      f"(retrying)", flush=True)
                continue
            raise
        if first_batch_time is None:
            first_batch_time = dt_b
            print(f"  throughput: {nb / dt_b:.1f} sets/s on {DEVICE} "
                  f"-> ~{dt_b / nb * total:.0f} s for {total} sets",
                  flush=True)
        for j, s in enumerate(seeds_b):
            pv_j = pv[j]
            m = float(pv_j.mean())
            mi = int(np.argmax(pv_j))
            upd = pv_j > best_p
            if upd.any():
                wins += int(upd.sum())
                best_a[upd] = A[j][upd]
                best_p[upd] = pv_j[upd]
            if m > run_best_mean:
                run_best_mean, run_best_mean_seed = m, int(s)
            if pv_j[mi] > run_best_i:
                run_best_i, run_best_i_seed, run_best_i_h = float(pv_j[mi]), \
                    int(s), mi
            if m > at_mean:
                at_mean, at_mean_seed, at_mean_run = m, int(s), run_id
            if pv_j[mi] > at_i:
                at_i, at_i_seed, at_i_h, at_i_run = float(pv_j[mi]), int(s), \
                    mi, run_id
            run_rows.append((int(s), fams_b[j], m, float(pv_j[mi]), mi))
            n_done += 1
            dt = time.time() - t0
            print(f"[{n_done}/{total}] ({100.0 * n_done / total:5.1f}%) "
                  f"run#{run_id} seed={s} [{fams_b[j]}] mean={m:.5f} "
                  f"max_inst={pv_j[mi]:.4f} @h{mi} | "
                  f"RUN BEST set {run_best_mean:.5f} (seed {run_best_mean_seed}) "
                  f"inst {run_best_i:.4f} (seed {run_best_i_seed}) | "
                  f"ALLTIME set {at_mean:.5f} (seed {at_mean_seed}, run #{at_mean_run}) "
                  f"inst {at_i:.4f} @h{at_i_h} (seed {at_i_seed}, run #{at_i_run}) "
                  f"| eta={dt / n_done * (total - n_done):.0f}s", flush=True)
        b0 += nb
    seen_f.close()

    # ---- persistent ledger of ALL seeds + TOP-1000 ---------------------
    merged_all = dict(prev_all)
    for s, fam, m, miv, mh in run_rows:
        merged_all[int(s)] = (int(s), fam, m, miv, mh, run_id)
    all_sorted = sorted(merged_all.values(), key=lambda r: -r[2])
    led_fmts = ["%d", "%s", "%.9f", "%.9f", "%d", "%d"]
    write_csv(all_p, "seed,family,set_mean,max_inst,best_h,run_id",
              all_sorted, led_fmts)
    write_csv(top_p, "seed,family,set_mean,max_inst,best_h,run_id",
              all_sorted[:1000], led_fmts)
    print(f"ledger: {len(all_sorted)} unique seeds total -> {all_p}; "
          f"TOP-1000 -> {top_p}", flush=True)

    # ---- outputs -------------------------------------------------------
    cols = (["id"] + [f"gamma_{k}" for k in range(P)]
            + [f"beta_{k}" for k in range(P)])
    out = np.concatenate([np.arange(H)[:, None], best_a], axis=1)
    np.savetxt(rat_p, out, delimiter=",", header=",".join(cols), comments="",
               fmt=["%d"] + ["%.12f"] * (2 * P))

    write_csv(f"{args.prefix}_report.csv",
              "seed,family,set_mean,max_inst,best_h", run_rows,
              ["%d", "%s", "%.9f", "%.9f", "%d"])

    hist_rows = [[r["run_id"], r["ts"], r["sets"], r["base_seed"],
                  r["best_mean"], r["best_seed"], r["best_inst"], r["best_h"],
                  r["best_inst_seed"]] for r in hist]
    hist_rows.append([run_id, int(time.time()), total, base,
                      run_best_mean, run_best_mean_seed, run_best_i,
                      run_best_i_h, run_best_i_seed])
    write_csv(hist_p,
              "run_id,ts,sets,base_seed,best_mean,best_seed,"
              "best_inst,best_h,best_inst_seed", hist_rows,
              ["%d", "%d", "%d", "%d", "%.9f", "%d", "%.9f", "%d", "%d"])

    top = sorted(run_rows, key=lambda r: -r[2])[:max(1, args.save_top)]
    nruns = len(hist) + 1
    nsets_total = sum(int(r["sets"]) for r in hist) + total
    print("=" * 78)
    print(f"LOTTERY DONE run#{run_id}: {total}/{total} fresh seeds "
          f"({len(seen)} total ever), {time.time() - t0:.1f} s on {DEVICE}, "
          f"wins vs ref this run: {wins}")
    print(f"  run best set : {run_best_mean:.5f} (seed {run_best_mean_seed})")
    print(f"  run best inst: {run_best_i:.4f} @h{run_best_i_h} "
          f"(seed {run_best_i_seed})")
    print(f"ALLTIME over {nruns} runs ({nsets_total} sets):")
    print(f"  best set-mean: {at_mean:.5f} (seed {at_mean_seed}, run #{at_mean_run})")
    print(f"  best inst:     {at_i:.4f} @h{at_i_h} (seed {at_i_seed}, "
          f"run #{at_i_run})")
    print("-" * 78)
    print(f"  {'seed':>12}  {'family':<9} {'set_mean':>9} {'max_inst':>9} best_h")
    for s, fam, m, miv, mh in top:
        print(f"  {s:>12}  {fam:<9} {m:>9.5f} {miv:>9.4f} h{mh}")
    print("-" * 78)
    print(f"  ALLTIME ledger top-3 (all runs):")
    for s, fam, m, miv, mh, rid in all_sorted[:3]:
        print(f"    seed {s:>12}  {fam:<9} set_mean {m:>9.5f}  "
              f"inst {miv:>7.4f} @h{mh}  (run #{rid})")
    print(f"  per-instance ratchet (all runs): {rat_p}")
    print(f"  run report (seed->result):       {args.prefix}_report.csv")
    print(f"  history (all runs):              {hist_p}")
    print(f"  full ledger (all seeds, all):    {all_p} ({len(all_sorted)})")
    print(f"  TOP-1000 seeds by set_mean:      {top_p}")
    print(f"  seen seeds (no repeats):         {seen_p} ({len(seen)})")
    print("=" * 78)


if __name__ == "__main__":
    main()