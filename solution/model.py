"""Сеть h -> углы QAOA (multi-output regression + multi-task aux-головы).

Архитектура:
  * общий ствол (MLP с GELU);
  * по отдельной голове на каждый из P=5 слоёв QAOA для gamma и для beta
    (углы предсказываются как «schedule» по слоям, а не 10 независимых
    выходов);
  * вспомогательные головы: энергия основного состояния (регрессия) и
    конфигурация основного состояния (12 сигмоид) — задают физический
    целевой сигнал, улучшают обобщение и делают модель объяснимой.

Выходы голов углов — линейные (без жёстких ограничений): сеть обучается
end-to-end через сам симулятор QAOA (loss = -P(ground)), поэтому диапазон
углов находит сама, а финальная «полировка» на конкретном инстансе
добирает остальное.
"""
import torch
import torch.nn as nn

N_QUBITS = 12


class QAOAAngleNet(nn.Module):
    def __init__(self, in_dim, hidden=(256, 256, 256), P=5, dropout=0.05):
        super().__init__()
        blocks = []
        d = in_dim
        for h in hidden:
            blocks += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*blocks)

        # по голове на слой QAOA
        self.g_heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(P)])
        self.b_heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(P)])

        # вспомогательные (multi-task) головы
        self.e_head = nn.Linear(d, 1)          # энергия основного состояния
        self.s_head = nn.Linear(d, N_QUBITS)   # конфигурация основного состояния (0/1)

    def _angles(self, t):
        g = torch.cat([head(t) for head in self.g_heads], dim=1)
        b = torch.cat([head(t) for head in self.b_heads], dim=1)
        return g, b

    def angles(self, x):
        """Только углы (для инференса)."""
        return self._angles(self.trunk(x))

    def forward(self, x):
        t = self.trunk(x)
        g, b = self._angles(t)
        e = self.e_head(t)
        s = torch.sigmoid(self.s_head(t))
        return g, b, e, s
