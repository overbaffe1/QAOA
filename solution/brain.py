"""QAOA «мозг» — персистентная память хороших угловых схем.

Прямой аналог ReactorBrain из MCTech (оптимизатор реактора):

    ReactorBrain.good_patterns     -> QAOABrain.patterns
                                      (лучшие (h, углы, P(ground)))
    ReactorBrain.rod_preferences   -> QAOABrain.preferences
                                      (среднее/ст.о. каждого из 10 углов
                                       по топ-схемам)
    ReactorBrain.top_schemes       -> QAOABrain.top_schemes
                                      (топ-5 глобальных схем)
    ReactorGenome.smart_randomize  -> QAOABrain.smart_randomize
    ReactorGenome.mutate           -> QAOABrain.mutate
    ReactorGenome.crossover        -> QAOABrain.crossover

Память сохраняется в brain.pkl МЕЖДУ запусками: после самопроверки на
h_train мозг помнит лучшие схемы и их статистику, а следующий запуск
(включая финальный на h_test) начинается «умной инициализацией» —
не с нуля, а от накопленного опыта.
"""
import os
import pickle

import numpy as np

N_ANGLES = 10          # 5 gamma + 5 beta
PATTERN_KEEP = 512     # сколько схем держать в мозге
TOP_N = 64             # по скольким топ-схемам считать статистику
TOP_SCHEMES = 5        # аналог ReactorBrain.top_schemes


class QAOABrain:
    """«Мозг», переживающий перезапуски (аналог brain.pkl в MCTech)."""

    def __init__(self):
        self.patterns = []      # [{'h': (12,), 'angles': (10,), 'score': float}]
        self.preferences = {}   # i -> (mean, std)
        self.top_schemes = []   # топ-5 глобальных схем, (10,)

    # ------------------------------------------------------------------ #
    # «Обновить память» — вызывается после прогона (update_* из MCTech)  #
    # ------------------------------------------------------------------ #
    def update_from_results(self, results, keep=PATTERN_KEEP):
        for h, angles, score in results:
            h = np.asarray(h, dtype=np.float64)
            angles = np.asarray(angles, dtype=np.float64)
            # для этого h уже есть не худшая схема — не дублируем
            if any(np.allclose(p["h"], h, atol=1e-6) and p["score"] >= score
                   for p in self.patterns):
                continue
            self.patterns.append(
                {"h": h, "angles": angles, "score": float(score)})
        self.patterns.sort(key=lambda p: -p["score"])
        self.patterns = self.patterns[:keep]
        top = np.stack([p["angles"] for p in self.patterns[:TOP_N]])
        self.preferences = {
            i: (float(top[:, i].mean()), float(max(top[:, i].std(), 1e-3)))
            for i in range(N_ANGLES)
        }
        self.top_schemes = [p["angles"].copy()
                            for p in self.patterns[:TOP_SCHEMES]]

    # ------------------------------------------------------------------ #
    # «Известная схема» — если этот h мы уже решали                       #
    # ------------------------------------------------------------------ #
    def best_for(self, h, tol=1e-6):
        for p in self.patterns:
            if np.allclose(p["h"], h, atol=tol):
                return p["angles"]
        return None

    # ------------------------------------------------------------------ #
    # «Умная инициализация» (smart_randomize из генома MCTech)            #
    # ------------------------------------------------------------------ #
    def smart_randomize(self, rng, net_angles):
        """30% — от топ-схем, 20% — от статистики углов,
        30% — предсказание сети + шум, 20% — с нуля (как в MCTech)."""
        u = rng.random()
        if self.top_schemes and u < 0.30:
            a = self.top_schemes[int(rng.integers(len(self.top_schemes)))]
            return a + rng.normal(0.0, 0.05, N_ANGLES)
        if self.preferences and u < 0.50:
            return np.array([rng.normal(self.preferences[i][0],
                                        self.preferences[i][1])
                             for i in range(N_ANGLES)])
        if u < 0.80:
            return np.asarray(net_angles, dtype=np.float64) \
                + rng.normal(0.0, 0.15, N_ANGLES)
        return self.fresh(rng)

    @staticmethod
    def fresh(rng):
        """Совсем новая схема: gamma ~ U[0, 2pi], beta ~ U[0, pi]."""
        a = np.empty(N_ANGLES)
        a[:5] = rng.uniform(0.0, 2.0 * np.pi, 5)
        a[5:] = rng.uniform(0.0, np.pi, 5)
        return a

    # ------------------------------------------------------------------ #
    # «Мутация» и «скрещивание» (как в ReactorGenome)                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def mutate(a, rng, strength=0.15):
        return np.asarray(a, dtype=np.float64) \
            + rng.normal(0.0, strength, N_ANGLES)

    @staticmethod
    def crossover(a, b, rng):
        mask = rng.integers(2, N_ANGLES).astype(bool)
        return np.where(mask, a, b)

    # ------------------------------------------------------------------ #
    def save(self, path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        if not os.path.exists(path):
            return QAOABrain()
        try:
            with open(path, "rb") as f:
                b = pickle.load(f)
            return b if isinstance(b, QAOABrain) else QAOABrain()
        except Exception:
            return QAOABrain()

    def __len__(self):
        return len(self.patterns)
