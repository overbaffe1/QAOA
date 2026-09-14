"""Smoke-тест solution.ipynb: исполняем все code-ячейки в одном namespace
с уменьшенными параметрами, чтобы проверить «одну кнопку» без полного прогона.

Расчитан на машину с малой RAM (песочница Arena: 3.9 ГБ, 2 CPU, без CUDA):
метки считаются кусками по 64 инстансов, полировка — на 32 инстансах.
В Colab (GPU) полный прогон делает сам ноутбук, этот тест там не нужен.

Запуск:  pip install nbformat && python smoke_test_nb.py
"""
import os
import shutil
import tempfile

import nbformat  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
nb = nbformat.read(os.path.join(ROOT, "solution.ipynb"), as_version=4)

workdir = tempfile.mkdtemp(prefix="qaoa_nb_")
shutil.copy(os.path.join(ROOT, "h_train.npy"), workdir)
shutil.copy(os.path.join(ROOT, "J.npy"), workdir)

# (что ищем, на что меняем) — если строка не найдена, ноутбук разъехался
# с этим тестом: сообщаем громко, иначе смоук-тест молча гоняет полный прогон.
REPLACEMENTS = [
    ("N_SYNTH_TRAIN, N_SYNTH_VAL, N_VAL_REAL = 1000, 500, 50",
     "N_SYNTH_TRAIN, N_SYNTH_VAL, N_VAL_REAL = 64, 64, 50"),
    ("BATCH, STEPS, LR, AUX_W = 128, 12000, 1e-3, 0.05",
     "BATCH, STEPS, LR, AUX_W = 32, 300, 1e-3, 0.05"),
    ("run(steps=250, restarts=3, lr=0.05, seed=SEED, chunk=CHUNK_LABELS)",
     "run(steps=40, restarts=1, lr=0.05, seed=SEED, chunk=64)"),
    ("POLISH_RESTARTS, POLISH_STEPS = 5, 250",
     "POLISH_RESTARTS, POLISH_STEPS = 2, 10"),
    ("POLISH_FINE_STEPS, POLISH_FINE_LR, POLISH_LR = 100, 0.01, 0.05",
     "POLISH_FINE_STEPS, POLISH_FINE_LR, POLISH_LR = 0, 0.01, 0.05"),
    # полировка — на 32 инстансах (батч 500 на backward не влезает в 3.9 ГБ)
    ("if h_test is not None:",
     "h_train = h_train[:32]\nif h_test is not None:"),
]

ns = {"__name__": "__main__"}


class FakeIP:
    def system(self, cmd):
        print("  [pip]", cmd)


ns["get_ipython"] = lambda: FakeIP()

all_code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
stale = [old for old, _ in REPLACEMENTS if old not in all_code]
if stale:
    print("ВНИМАНИЕ: ноутбук разъехался со смоук-тестом, не найдены строки:")
    for s in stale:
        print("   ", s)
    print("(обновите REPLACEMENTS — иначе тест молча гоняет полный прогон)\n")

os.chdir(workdir)
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = cell.source
    for old, new in REPLACEMENTS:
        src = src.replace(old, new)
    print(f"===== cell {i} =====", flush=True)
    exec(compile(src, f"cell{i}", "exec"), ns)

print("\nSMOKE-ТЕСТ ПРОЙДЕН")
print("artifacts:", os.listdir(os.path.join(workdir, "data")))
sub = os.path.join(workdir, "submission.csv")
if os.path.exists(sub):
    with open(sub, encoding="utf-8") as f:
        rows = f.read().strip().splitlines()
    print(f"submission.csv: {len(rows) - 1} строк, заголовок: {rows[0]}")
    [[float(c) for c in r.split(",")] for r in rows[1:]]
    print("все ячейки читаются float() — ОК")
else:
    print("submission.csv не создан (нет h_test.npy — это ожидаемо)")
