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

# --- инференс ---
POLISH_STEPS = 40          # шагов "полировки" Adam на инстанс от выхода сети
POLISH_LR = 0.05
