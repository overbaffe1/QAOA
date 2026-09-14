# HANDOFF — продолжение работы для нового агента

> Прочти это ПОЛНОСТЬЮ перед началом. Это сжатая память предыдущих сессий:
> задача, текущее состояние, план, грабли.

## Задача (олимпиада РКЦ, «Машинное обучение в квантовых вычислениях»)

Дана фиксированная `J.npy` (12×12) и вектора `h` (12,). Для каждого `h`
предсказать углы QAOA глубины p=5 — `gamma_0..4`, `beta_0..4` —
максимизирующие `P(ground)`. Метрика: среднее P(ground) по 500 тестовым `h`.
`P(ground)` = сумма вероятностей всех состояний с минимальной энергией
(дифференцируемый симулятор — `QAOA.py` из условия, не менять).

**Правила:**
- Файл посылки: `submission.csv` — 500 строк: `id, gamma_0..4, beta_0..4`.
  Значения должны читаться `float()` — только числа, без префиксов!
- Финал: `h_test.npy` выдан **15.09.2026 в 18:00 МСК**; инференс на нём —
  **лимит 10 минут**. Финальная оценка — P(ground) на h_test.
- Публичный лидерборд (h_train): **сырое P(ground), без нормализации,
  БЕЗ лимита 10 минут** (подтверждено: наша посылка 0.2019234299659729 =
  ровно среднее из CSV). Топ лидерборда 0.95191 — это тяжёлый оффлайн-
  брутфорс h_train.
- Очки: ранговые, Score_i = 70·(1−(i−1)/n) (+ до 20 за ML-качество,
  + до 10 за презентацию/воспроизводимость). Подача: submission.csv +
  solution.ipynb + presentation.pdf + README; ноутбук должен работать
  в Colab «в одну кнопку».

## Текущие цифры (прогоны на GPU пользователя)

| что | P(ground), mean |
|---|---|
| метки: 3×250 Adam по 500 h_train | 0.2063 (median 0.1813, max 0.893, min 0.005) |
| чистая сеть (end-to-end обученная), валидация | 0.1972 |
| чистая сеть, самопроверка 500 h_train | 0.1890 |
| **full профиль (8×300+100) — было на лидерборде** | **0.2019** (435 с GPU) |
| brain профиль (8×2×150+100+L-BFGS, дефолт) | ожидается ~0.202–0.21, ~8–9 мин GPU |
| потолок: 24×700 Adam, все 500 | 0.2823 |
| потолок: 64×500+L-BFGS / 128×220+top-8 (по 8 инстансам) | ~0.31 / max ~0.30 |

**Ключевой инсайт (лог полного прогона пользователя):** сеть уже в
правильном бассейне; 6 случайных рестартов дали всего +0.0009. Не тратить
бюджет на случайные рестарты — только «умные» (сеть, память «мозга»,
скрещивания/мутации от лучших).

## Состояние на 14.09.2026, сессия `arena/01a0a212-qaoa`

1. **Фикс индексов `polish_brain` (локальный коммит `5d61898`) УЖЕ В `main`**
   — `solution/infer.py`, строки с `order = np.argsort(-fit, axis=0)`:
   `t1/t2 = np.stack([opt10[c][i] for i, c in enumerate(order[0/1])])`.
   Допушивать нечего. Проверено вживую:
   * прогон `--profile brain --limit 8 --pop 3` (B > pop — условие краша)
     проходит оба поколения и пишет CSV;
   * та же команда на коде ДО фикса (временно пропатченная копия) падает
     `IndexError: index 3 is out of bounds for axis 0 with size 3` ровно в
     конце поколения 1 — как и было описано.
   Заодно добавлена защита на случай, если выжило меньше двух кандидатов.
2. **`submission.csv` в гите был ЗАГЛУШКОЙ** (случайные `np.random.rand`,
   P(ground) = 0.00209), хотя на лидерборд загружался другой файл (0.20192)
   с машины пользователя. Распаковка ZIP из GitHub поверх рабочей папки
   молча подменяла хорошую посылку заглушкой. Заменён на настоящую посылку,
   посчитанную в этой сессии (см. «Что сделано» ниже и таблицу результатов).
   **Перед любой загрузкой на платформу:**
   `python solution/evaluate.py --csv submission.csv --h h_train.npy` —
   число должно совпадать с лидербордом.
3. **Новые инструменты** (все проверены прогонами):
   * `solution/heavy_rounds.py` — тяжёлые раунды на h_train пачкой: сам
     запускает `infer.py --profile brain` с разными сидами, пишет логи в
     `data/heavy/`, оценивает каждый CSV, копирует лучший в
     `submission_best.csv`, переживает Ctrl-C, `--skip-existing`;
   * `solution/polish_csv.py` — доводка готовой посылки/меток
     (per-instance L-BFGS + опциональный батчевый Adam `--fine`),
     «keep best» — ухудшить не может, работает на CPU;
   * `evaluate.py --csv submission.csv` — сырое среднее P(ground) готовой
     посылки (ровно то, что видит лидерборд), плюс проверка заголовка,
     id и того, что все ячейки читаются `float()`;
   * `infer.py --time-budget N` — жёсткий бюджет секунд (этап, который не
     успевает, не начинается; L-BFGS прерывается на инстансе; CSV пишется
     всегда). Для h_test: `--time-budget 520`;
   * `generate_labels.py --chunk N` — метки кусками по N инстансов
     (эквивалентно по результату, память ∝ N): без этого батч 500 на
     backward в песочнице убивается OOM-киллером.

## Что сделано в этой сессии (кроме проверок)

* Прогнана честная генерация меток на CPU песочницы:
  `generate_labels.py --steps 250 --restarts 3 --chunk 64 --seed 42`
  (~40 мин, peak RSS ~1.2 ГБ) → `data/labels.npz` (в гит не попадает).
* Из меток собрана настоящая `submission.csv` (+ доводка `polish_csv.py`),
  результат измерен `evaluate.py --csv` — см. таблицу результатов и лог
  коммита. Это сразу лучше заглушки (0.00209) и не хуже лидерборда (0.20192).

## Код (всё в `main`)

```
solution/
  brain.py            # QAOABrain — персистентная память (аналог ReactorBrain
                      # из Minecraft-кода пользователя): patterns (лучшие
                      # (h, углы, P)), preferences (mean/σ по 10 углам),
                      # top_schemes (топ-5), smart_randomize/mutate/crossover
  infer.py            # профили: full (8×300+100), fast (3×100), brain
                      # (популяция pop×gens×steps + fine + L-BFGS-доводка);
                      # CLI: --profile --pop --gens --steps --fine --restarts
                      #       --seed --no-refine --refine-iters --limit
                      #       --time-budget
                      # память: data/brain.pkl (накапливается между запусками)
  heavy_rounds.py     # пачка тяжёлых раундов + выбор лучшего CSV
  polish_csv.py       # доводка готовой посылки (L-BFGS, keep best)
  config.py, features.py, model.py, generate_labels.py (--chunk), train.py,
  evaluate.py (--csv), build_notebook.py, run_pipeline.sh
solution.ipynb        # Colab «в одну кнопку» (5×250+100 для T4)
presentation.pdf, README.md   # с реальными цифрами
data/                 # АРТЕФАКТЫ (в гит не попадают!): model.pt, brain.pkl,
                      # labels.npz, heavy/ — у пользователя в
                      # C:\Users\Дракула\QAOA\data\
```

## План (что осталось сделать)

1. **Тяжёлые оффлайн-раунды** на h_train (растит публичный лидерборд),
   на GPU пользователя — в песочнице нет CUDA и 2 CPU, там это дни:
   ```
   cd solution
   python heavy_rounds.py --rounds 3 --start-seed 7
   ```
   (внутри: `infer.py --h h_train.npy --profile brain --pop 12 --gens 3
   --steps 400 --refine-iters 50 --seed 7/8/9`, ~40–45 мин на раунд).
   Результат каждого раунда ≥ 0.2019 → лучший (`submission_best.csv`)
   загрузить на лидерборд. `brain.pkl` накапливает схемы, каждый раунд
   стартует умнее («мозг между запусками», как в Minecraft-коде).
   Дёшево добить остаток: `python polish_csv.py --csv ../submission_best.csv
   --h ../h_train.npy --out ../submission_best2.csv --refine-iters 50`.
2. **Финал (после 18:00 МСК 15.09):** скачать `h_test.npy` в корень →
   ```
   python infer.py --h h_test.npy --out submission_full.csv  --profile full  --time-budget 520
   python infer.py --h h_test.npy --out submission_brain.csv --profile brain --time-budget 520
   python evaluate.py --csv submission_full.csv  --h h_test.npy   # если времени в обрез — пропустить
   ```
   на платформу — лучший из full/brain. `--time-budget 520` гарантирует, что
   CSV запишется даже если доводка не успеет (лимит 600 с). brain.pkl с
   h_train-схемами для h_test не даст точных совпадений, но даст разнообразие
   инициализаций (статистика углов, топ-схемы).
3. При желании — Colab-прогон ноутбука как запасной путь для финала.

## Контекст пользователя

- Windows, `C:\Users\Дракула\QAOA\solution`, GPU CUDA (RTX), Python.
- **Работает через ZIP-архивы с GitHub, НЕ через git clone** — в его папке
  смесь клона и ZIP, `git pull` даёт конфликты. `data/` в ZIP нет —
  при распаковке архива НЕ трогать `data/`. **`submission.csv` в ZIP есть —
  распаковка поверх папки затирает локальную посылку** (см. выше).
- Его предыдущий проект: оптимизатор реактора для Minecraft (MCTech) —
  генетический алгоритм с персистентным «мозгом» (brain.pkl: good_patterns,
  rod_preferences, top_schemes; smart_randomize/mutate/crossover; «final
  refinement»). Профиль `brain` в infer.py — прямой порт его подхода;
  пользователь мыслит в этих категориях, отвечать в этих терминах.
- Язык общения — русский. Сроки: h_test в 18:00 МСК 15.09.2026.

## Грабли (не наступать повторно)

- **Сессии Arena закрываются после мерджа PR** — пуш больше невозможен,
  нужен новый coding-сессию. Рабочая область (workspace) переживает
  сессию, но между ТУРНАМИ её обнуляли 3 раза (venv, data/, HEAD) —
  чекать `git status`/процессы/venv в начале каждого хода; коммитить и
  пушить часто; git remote — единственный надёжный источник истины.
- **CSV писать `fmt='%.12f'`, НИКОГДА не `%r`** — в NumPy 2.x
  `repr(np.float64)` = `np.float64(...)` и платформа падает на парсинге
  (это уже было: посылка не засчиталась, чинили regex'ом).
- **Песочница Arena: 3.9 GB RAM / 2 CPU, CUDA нет.** Батч 500×4096×p=5 на
  backward = OOM-килл сразу → `generate_labels.py --chunk 64` (peak ~1.2 ГБ);
  `polish_csv.py` per-instance — лёгкий. Тяжёлые раунды тут не идут —
  только на GPU пользователя.
- **`download.pytorch.org` в песочнице закрыт** (TLS EOF). torch ставится с
  PyPI: `python3 -m venv /home/user/.venv` → `pip install numpy` →
  `pip install --no-deps torch==2.9.1` → `pip install typing_extensions
  filelock sympy networkx jinja2 fsspec` → `pip install --no-deps
  nvidia-cuda-runtime-cu12==12.8.90 nvidia-cublas-cu12==12.8.4.1
  nvidia-nvjitlink-cu12==12.8.93 nvidia-cudnn-cu12==9.10.2.21
  nvidia-cusparselt-cu12==0.7.1 nvidia-cufft-cu12==11.3.3.83
  nvidia-curand-cu12==10.3.9.90 nvidia-cusolver-cu12==11.7.3.90
  nvidia-cusparse-cu12==12.5.8.93 nvidia-cuda-nvrtc-cu12==12.8.93
  nvidia-cuda-cupti-cu12==12.8.90 nvidia-nvtx-cu12==12.8.90
  nvidia-nccl-cu12==2.27.5 nvidia-cufile-cu12==1.13.1.3
  nvidia-nvshmem-cu12==3.3.20` (без nvidia-библиотек `import torch` падает;
  `torch.cuda.is_available()` = False, CPU-опы работают).
- **`data/model.pt` в песочнице нет** (в гит не кладётся). Для смоук-тестов
  `infer.py` кладётся случайный чекпоинт (сеть с случайными весами даёт
  P≈0.0002, но полировка всё равно вытягивает ~0.22 на 8 инстансах) — и
  **удаляется после тестов**; так же удалять `data/brain.pkl`, набитый
  схемами из смоук-тестов со случайной сетью, иначе «мозг» будет мусорным.
- `build_notebook.py`: исходники ячеек внутри `code("""...""")` —
  тройные кавычки внутри запрещены.
- `torch.where` для per-instance best-of маски: нужно `.unsqueeze(1)`.
- Потолок-эксперименты с чистыми популяциями (случайные/большинство) —
  МЁРТВАЯ СОБСТВЕННОСТЬ, повторять бессмысленно: см. таблицу выше.
- Лимит 10 минут — только для h_test; лидерборд (h_train) можно
  брутфорсить оффлайн сколько угодно.
- Если в песочнице нет `data/model.pt` — pipeline `run_pipeline.sh`
  генерирует (labels ~40 мин CPU → train ~3 ч CPU).
