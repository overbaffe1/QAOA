"""Smoke-тест solution.ipynb: исполняем все code-ячейки в одном namespace
с уменьшенными параметрами, чтобы проверить «одну кнопку» без полного прогона.
"""
import json
import os
import shutil
import sys
import tempfile

import nbformat  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
nb = nbformat.read(os.path.join(ROOT, "solution.ipynb"), as_version=4)

workdir = tempfile.mkdtemp(prefix="qaoa_nb_")
shutil.copy(os.path.join(ROOT, "h_train.npy"), workdir)
shutil.copy(os.path.join(ROOT, "J.npy"), workdir)

ns = {"__name__": "__main__"}
import builtins


class FakeIP:
    def system(self, cmd):
        print("  [pip]", cmd)


ns["get_ipython"] = lambda: FakeIP()

os.chdir(workdir)
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = cell.source
    src = src.replace("BATCH, STEPS, LR, AUX_W = 128, 12000, 1e-3, 0.05",
                      "BATCH, STEPS, LR, AUX_W = 32, 300, 1e-3, 0.05")
    src = src.replace("N_SYNTH_TRAIN, N_SYNTH_VAL, N_VAL_REAL = 1000, 500, 50",
                      "N_SYNTH_TRAIN, N_SYNTH_VAL, N_VAL_REAL = 64, 64, 50")
    src = src.replace("run(steps=250, restarts=3, lr=0.05, seed=SEED)",
                      "run(steps=40, restarts=1, lr=0.05, seed=SEED)")
    src = src.replace("POLISH_STEPS, POLISH_LR = 40, 0.05",
                      "POLISH_STEPS, POLISH_LR = 10, 0.05")
    print(f"===== cell {i} =====", flush=True)
    exec(compile(src, f"cell{i}", "exec"), ns)

print("\nSMOKE-ТЕСТ ПРОЙДЕН")
print("artifacts:", os.listdir(os.path.join(workdir, "data")))
