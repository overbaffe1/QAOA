"""Константы и гиперпараметры пайплайна."""
import os

SEED = 42
P = 5                      # глубина QAOA (число углов каждого типа)
N_QUBITS = 12
DIM = 2 ** N_QUBITS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                # корень репозитория: J.npy, h_train.npy
DATA = os.path.join(ROOT, "data")           # артефакты: метки, модель
os.makedirs(DATA, exist_ok=True)

# --- генерация меток ---
LABEL_STEPS = 250          # шагов Adam на рестарт
LABEL_RESTARTS = 3         # количество случайных инициализаций
LABEL_LR = 0.05

# --- обучение ---
N_SYNTH_TRAIN = 1000       # синтетические h (uniform [-1,1]^12) для дообучения
N_SYNTH_VAL = 500          # синтетические h для валидации
N_VAL_REAL = 50            # holdout из реальных 500
BATCH = 128
STEPS = 12000
LR = 1e-3
AUX_W = 0.05               # вес вспомогательных (ground-state) потерь

# --- инференс: сильная поинстансная полировка ---
# Рестарт 0 — предсказание сети, рестарт 1 — предсказание + пертурбация,
# остальные — случайные. Для каждого инстанса оставляем лучший набор углов.
POLISH_RESTARTS = 8          # full-профиль (GPU/Colab)
POLISH_STEPS = 300           # шагов Adam на рестарт (lr=POLISH_LR)
POLISH_FINE_STEPS = 100      # финальная доводка мелким lr от лучшего
POLISH_FINE_LR = 0.01
POLISH_LR = 0.05
# fast-профиль (локальная самопроверка на CPU)
POLISH_RESTARTS_FAST = 3
POLISH_STEPS_FAST = 100
POLISH_FINE_STEPS_FAST = 0
