"""Физические признаки для сети.

Ключевая идея: матрица J фиксирована и имеет точную зеркальную симметрию
i <-> 11-i, поэтому вводим симметризованную/антисимметризованную по этой
зеркальной операции проекции h, плюс точные (получаемые перебором всех
2^12 = 4096 конфигураций) характеристики основного состояния модели Изинга:
энергия, зазор, конфигурация, вырожденность, согласованность h с основным
состоянием.
"""
import numpy as np

N_QUBITS = 12
DIM = 2 ** N_QUBITS


def _s_matrix():
    """(4096, 12) матрица конфигураций s in {+1, -1}^12,
    порядок битов совпадает с расшифровкой индексов в QAOA.py
    (старший бит -> кубит 0)."""
    bits = np.arange(DIM)
    S = 2 * ((bits[:, None] >> np.arange(N_QUBITS - 1, -1, -1)) & 1) - 1
    return S


_S = _s_matrix()


def ground_state_data(J, h):
    """Точные характеристики основного состояния для каждого h.

    h: (M, 12). Конвенция энергии совпадает с QAOA.py:
    E(s) = 0.5 * s J s^T + h s.
    Возвращает dict: Emin, gap, s_star (M,12) in {-1,+1}, deg.
    """
    h = np.atleast_2d(np.asarray(h, dtype=np.float64))
    E = 0.5 * np.einsum("ij,ki,kj->k", J, _S, _S) + h @ _S.T   # (M, 4096)
    order = np.argsort(E, axis=1, kind="stable")
    rows = np.arange(h.shape[0])[:, None]
    Emin = E[rows, order[:, :1]].squeeze(1)
    E2 = E[rows, order[:, 1:2]].squeeze(1)
    gap = E2 - Emin
    s_star = _S[order[:, 0]]
    deg = (E <= Emin[:, None] + 1e-9).sum(axis=1)
    return {"Emin": Emin, "gap": gap, "s_star": s_star, "deg": deg}


def build_features(J, h):
    """Входные признаки для сети.

    Возвращает (X, gs): X — (M, D) признаки; gs — ground_state_data.
    """
    h = np.atleast_2d(np.asarray(h, dtype=np.float64))
    gs = ground_state_data(J, h)
    s = gs["s_star"]

    # зеркальная симметрия J: i <-> 11-i
    h_mirror = h[:, ::-1]
    h_sym = 0.5 * (h + h_mirror)
    h_asym = 0.5 * (h - h_mirror)

    phys = np.stack(
        [
            np.einsum("ij,ij->i", h, s),  # согласованность h с основным состоянием
            np.linalg.norm(h, axis=1), # амплитуда поля
            np.abs(h).mean(axis=1),
            h.std(axis=1),
            gs["Emin"],               # энергия основного состояния
            gs["gap"],                # зазор до первого возбуждённого
        ],
        axis=1,
    )
    X = np.concatenate([h, h_sym, h_asym, phys], axis=1).astype(np.float32)
    return X, gs
