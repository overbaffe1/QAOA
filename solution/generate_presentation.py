"""Генерация presentation.pdf (A4 landscape, DejaVu Sans).

Читает актуальные цифры из data/results.json (если есть); иначе — плейсхолдеры.
Запуск: python generate_presentation.py
"""
import json
import os

from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FONT_DIR = "/usr/share/fonts/truetype/dejavu"

NAVY = (18, 36, 64)
ACCENT = (0, 113, 188)
GREY = (90, 96, 104)
LIGHT = (240, 244, 249)


def get_results():
    """data/results.json (свежие прогоны) -> ROOT/results.json (в гите)."""
    for path in (os.path.join(ROOT, "data", "results.json"),
                 os.path.join(ROOT, "results.json")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return {}


R = get_results()


def fmt(key, nd=4, suffix=""):
    v = R.get(key)
    if v is None:
        return "—"
    return f"{v:.{nd}f}{suffix}"


class Deck(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("DejaVu", "", 8)
            self.set_text_color(*GREY)
            self.cell(0, 6, "QAOA углы через ML  ·  задача «Квантовый центр»",
                      align="L")
            self.set_draw_color(*ACCENT)
            self.set_line_width(0.6)
            self.line(10, self.get_y() + 1, self.epw - 10, self.get_y() + 1)
            self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 8, f"{self.page_no()}", align="R")

    def title_bar(self, num, text):
        self.set_fill_color(*NAVY)
        self.rect(10, 14, 9, 14, "F")
        self.set_xy(10, 14)
        self.set_font("DejaVu", "B", 13)
        self.set_text_color(255, 255, 255)
        self.cell(9, 14, str(num), align="C")
        self.set_xy(23, 15)
        self.set_font("DejaVu", "B", 17)
        self.set_text_color(*NAVY)
        self.multi_cell(self.epw - 15, 8, text)
        self.ln(4)

    def bullets(self, items, size=11, gap=3):
        for it in items:
            if isinstance(it, tuple):
                head, rest = it
                self.set_font("DejaVu", "B", size)
                self.set_text_color(*NAVY)
                self.cell(6, 6, "•")
                self.cell(self.get_string_width(head), 6, head)
                self.set_font("DejaVu", "", size)
                self.set_text_color(40, 40, 40)
                w = self.epw - 22
                self.multi_cell(w, 6, rest)
                self.ln(gap - 2)
            else:
                self.set_font("DejaVu", "", size)
                self.set_text_color(40, 40, 40)
                self.cell(6, 6, "•")
                self.multi_cell(self.epw - 22, 6, it)
                self.ln(gap)

    def formula_box(self, text, y=None):
        if y is not None:
            self.set_y(y)
        x = self.get_x()
        self.set_fill_color(*LIGHT)
        w = self.epw - 20
        self.set_font("DejaVu", "", 11)
        h = 10 * (text.count("\n") + 1) + 8
        self.rect(x, self.get_y(), w, h, "F")
        self.set_xy(x + 6, self.get_y() + 4)
        self.set_text_color(*NAVY)
        self.multi_cell(w - 12, 10, text)
        self.ln(6)


def new_slide(deck, num, title):
    deck.add_page()
    deck.title_bar(num, title)


pdf = Deck(orientation="L", format="A4")
pdf.set_margins(10, 10, 10)
pdf.add_font("DejaVu", "", f"{FONT_DIR}/DejaVuSans.ttf")
pdf.add_font("DejaVu", "B", f"{FONT_DIR}/DejaVuSans-Bold.ttf")

# ---- 1. титул ----
pdf.add_page()
pdf.set_fill_color(*NAVY)
pdf.rect(0, 60, 297, 90, "F")
pdf.set_xy(20, 78)
pdf.set_font("DejaVu", "B", 26)
pdf.set_text_color(255, 255, 255)
pdf.multi_cell(257, 12, "Оптимальные углы QAOA предсказанием\nмашинного обучения")
pdf.set_xy(20, 110)
pdf.set_font("DejaVu", "", 12)
pdf.set_text_color(190, 210, 235)
pdf.cell(0, 8, "12-кубитная модель Изинга · QAOA глубины p=5 · multi-output regression")
pdf.set_xy(20, 165)
pdf.set_font("DejaVu", "", 11)
pdf.set_text_color(*GREY)
pdf.cell(0, 7, "Задача «Машинное обучение в квантовых вычислениях» (РКЦ, Сколково)")
pdf.ln(7)
pdf.cell(0, 7, "Команда: <имя участника>")

# ---- 2. задача ----
new_slide(pdf, 1, "Задача")
pdf.bullets([
    ("Дано: ", "фиксированная матрица J (12x12), 500 векторов h_train (и h_test) "
     "линейных коэффициентов, дифференцируемый симулятор QAOA.py."),
    ("Предсказать: ", "для каждого h_test — углы (gamma_0..4, beta_0..4) схемы QAOA "
     "глубины 5, максимизирующие P(ground)."),
    ("Метрика: ", "P(ground) = вероятность основного состояния QAOA-состояния; "
     "баллы — по рангу среди воспроизводимых решений: Score = 70·(1-(i-1)/n)."),
    ("Ограничения: ", "Colab «в одну кнопку»; инференс ≤ 10 мин; углы обязаны "
     "зависеть от h (константные — 0 баллов)."),
])
pdf.formula_box("P(ground)_k = Σ_{s: E_k(s)=E_k^min} |⟨s|ψ_k(γ_k, β_k)⟩|²\n"
                "метрика = (1/500) Σ_k P(ground)_k")

# ---- 3. почему ML ----
new_slide(pdf, 2, "Почему ML: классический цикл — узкое место")
pdf.bullets([
    "Классический цикл QAOA ищет углы итеративно: тысячи запусков схемы на "
    "инстанс — вычислительное узкое место.",
    "ML-модель f(h) -> (gamma, beta) предсказывает близкие к оптимуму углы "
    "сразу: тысячи запусков заменяются одним проходом сети + десятки шагов "
    "доводки.",
    ("Идея решения: ", "гибрид «ML-тёплый старт + короткая классическая доводка»: "
     "сеть задаёт стартовые углы, а поинстансная доводка Adam (5 рестартов "
     "x 250 шагов + 100 шагов мелким lr, best-of по P(ground)) добирает "
     "остальное."),
    "Симулятор QAOA.py дифференцируем — можно обучать сеть сразу через P(ground).",
])

# ---- 4. наблюдения ----
new_slide(pdf, 3, "Ключевые наблюдения")
pdf.bullets([
    ("Симметрия J: ", "точное отражение i <-> 11-i — используем симметризованные "
     "и антисимметризованные проекции h."),
    ("Основное состояние: ", "получается точным перебором 2^12 = 4096 конфигураций "
     "(микросекунды); минимумы не вырождены."),
    ("Неуникальность углов: ", "в ландшафте P(ground) несколько локальных минимумов "
     "с сопоставимым качеством — L2-регрессия на «метки» усредняет по бассейнам "
     "и деградирует."),
    ("Вывод: ", "обучаем сеть end-to-end с потерей -P(ground) — сеть сама выбирает "
     "любой эквивалентный по качеству набор углов."),
])

# ---- 5. пайплайн ----
new_slide(pdf, 4, "Пайплайн")
pdf.set_font("DejaVu", "B", 12)
x = 12
labels = [
    ("1. Метки", "поинстансный Adam:\n3 рестарта x 250 шагов,\nбатч = все 500 h"),
    ("2. Признаки", "h + симметрии J +\nфизика (перебор 2^12):\nE_min, gap, h.s*, ..."),
    ("3. Обучение", "MLP, end-to-end:\nloss = -P(ground)+aux;\n450 real + 1000 synth"),
    ("4. Инференс", "f(h) -> углы +\n5 x 250 Adam,\nbest-of по h"),
]
for i, (t, d) in enumerate(labels):
    pdf.set_fill_color(*LIGHT if i % 2 else (224, 235, 247))
    pdf.rect(x, 45, 62, 60, "F")
    pdf.set_xy(x + 5, 50)
    pdf.set_font("DejaVu", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, t)
    pdf.set_xy(x + 5, 60)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(52, 6, d)
    if i < 3:
        pdf.set_xy(x + 62, 70)
        pdf.set_font("DejaVu", "B", 16)
        pdf.set_text_color(*ACCENT)
        pdf.cell(8, 8, "->")
    x += 70
pdf.set_xy(12, 120)
pdf.set_font("DejaVu", "", 10)
pdf.set_text_color(*GREY)
pdf.multi_cell(273, 6,
               "Валидация: 50 реальных h (holdout из 500) + 500 синтетических "
               "h ~ U[-1,1]^12 — защита от заучивания публичного лидерборда "
               "(он считается на h_train).")

# ---- 6. признаки ----
new_slide(pdf, 5, "Физические признаки")
pdf.bullets([
    ("h (12): ", "вектор линейных коэффициентов."),
    ("h_sym, h_asym (12+12): ", "проекции по зеркальной симметрии J — "
     "сеть «знает» симметрию задачи."),
    ("E_min, gap: ", "энергия основного состояния и зазор до первого "
     "возбуждённого (точный перебор 4096 конфигураций)."),
    ("h · s*: ", "согласованность линейного члена с основным состоянием."),
    ("||h||, mean|h|, std(h): ", "характеристика поля."),
    "Итог: 42 признака на инстанс; все вычисляются за миллисекунды без "
    "квантового симулятора.",
])

# ---- 7. архитектура ----
new_slide(pdf, 6, "Архитектура модели")
pdf.bullets([
    ("Ствол: ", "3 слоя x 256 нейронов, GELU, dropout 0.05."),
    ("Угловые головы: ", "по одной голове на каждый из 5 слоёв QAOA для gamma "
     "и для beta — углы предсказываются как «schedule» по глубине "
     "(10 выходов), а не как независимые числа."),
    ("Aux-головы (multi-task): ", "энергия основного состояния (MSE) и его "
     "конфигурация (BCE, 12 бит) — физический сигнал, улучшает обобщение."),
    ("Инициализация: ", "смещения углових голов — средние значения меток "
     "(полная поинстансная оптимизация)."),
])
pdf.formula_box("loss = - mean_k P(ground)(h_k, f(h_k))  +  0.05 · (MSE(E) + BCE(s*))\n"
                "обучение через дифференцируемый QAOA-симулятор из условия (без изменений)")

# ---- 8. инференс ----
new_slide(pdf, 7, "Инференс и «полировка»")
pdf.bullets([
    "Шаг 1: f(h) -> (gamma, beta) — один проход сети по 500 инстансам (мс).",
    ("Шаг 2 — полировка (профиль full): ", "5 рестартов по 250 шагов Adam "
     "(рестарт 0 — сеть, 1 — сеть + шум N(0, 0.3), 2..4 — случайные углы "
     "gamma ~ U[0, 2pi], beta ~ U[0, pi]) + 100 шагов доводки мелким lr; "
     "для каждого h остаётся лучший набор углов по P(ground)."),
    ("Альтернатива (профиль brain): ", "популяционный поиск с персистентной "
     "памятью data/brain.pkl — умная инициализация из накопленных схем, отбор, "
     "скрещивание/мутация, затем per-instance L-BFGS (keep best: результат "
     "принимается, только если P(ground) не стал хуже)."),
    "Тысячи запусков схемы -> сотни шагов Adam: сеть + полировка теряют доли "
     "процента P(ground) против полной поинстансной оптимизации.",
    ("Время: ", "на T4 (Colab) full ~5-8 мин, brain ~8-9 мин при лимите 10 мин; "
     "страховка --time-budget 520 — этап, который не успевает, не начинается, "
     "submission.csv записывается всегда. Детерминировано -> воспроизводимо."),
    "Углы детерминированно зависят от h — требованию «не константные углы» "
    "удовлетворяем конструктивно.",
])

# ---- 9. результаты ----
new_slide(pdf, 8, "Результаты")
rows = [
    ("полная поинстансная оптимизация (потолок, GPU)", fmt("label_mean"), fmt("label_median")),
    ("то же на CPU кусками по 64 (--chunk 64)", fmt("label_mean_cpu"), fmt("label_median_cpu")),
    ("посылка в репозитории: метки + L-BFGS 50", fmt("submission_repo", 6), fmt("submission_repo_median")),
    ("сеть (чистая, без полировки), holdout+synth", fmt("net_val"), ""),
    ("сеть + полировка, 50 реальных (holdout)", fmt("net_val_polished"), ""),
    ("самопроверка на h_train: сеть / сеть+полировка", fmt("selfcheck_net"), fmt("selfcheck_polished")),
    ("публичный лидерборд (full, 500 h_train)", fmt("leaderboard", 6), ""),
    ("лучший тяжёлый оффлайн-раунд (brain + L-BFGS, h_train)", fmt("heavy_best", 6), ""),
]
# прочерк — значит прогон ещё не выполнялся: такую строку не показываем
rows = [r for r in rows if "—" not in r[1]]
pdf.set_font("DejaVu", "B", 11)
pdf.set_fill_color(*NAVY)
pdf.cell(150, 9, "  конфигурация", fill=True)
pdf.cell(60, 9, "  mean P(ground)", fill=True)
pdf.cell(60, 9, "  median", fill=True)
pdf.ln()
pdf.set_font("DejaVu", "", 10.5)
for i, (name, a, b) in enumerate(rows):
    fill = LIGHT if i % 2 else (255, 255, 255)
    pdf.set_fill_color(*fill)
    pdf.cell(150, 9, f"  {name}", fill=True)
    pdf.cell(60, 9, f"  {a}", fill=True)
    pdf.cell(60, 9, f"  {b}", fill=True)
    pdf.ln()
pdf.ln(4)
pdf.set_font("DejaVu", "", 10)
pdf.set_text_color(*GREY)
pdf.multi_cell(273, 6,
               f"Время инференса (500 h, сеть + полировка): {R.get('time_infer', '—')} с "
               "(лимит 600 с).\nВсе сиды фиксированы (SEED=42); чекпоинты: "
               "data/labels.npz, data/model.pt, data/history.csv.")

# ---- 10. воспроизводимость ----
new_slide(pdf, 9, "Воспроизводимость и подача")
pdf.bullets([
    ("Colab «в одну кнопку»: ", "solution.ipynb самодостаточен — зависимости, J, "
     "h_train и QAOA-класс встроены; нужно только приложить h_test.npy и "
     "выполнить Run all."),
    ("Локально: ", "pip install -r requirements.txt; "
     "generate_labels.py (--chunk 64 при малой RAM) -> train.py -> infer.py; "
     "heavy_rounds.py — тяжёлые раунды на h_train, polish_csv.py — доводка "
     "готовой посылки, evaluate.py --csv — сверка числа с лидербордом."),
    ("Версии: ", "numpy, torch (в Colab установлен); все сиды SEED=42."),
    ("Чекпоинты: ", "метки после каждого рестарта, лучшая модель по валидации, "
     "история обучения — повторный запуск продолжает с чекпоинтов."),
    ("Структура: ", "README.md (описание и запуск), requirements.txt, "
     "solution/ (код), solution.ipynb (подача), presentation.pdf."),
])

out = os.path.join(ROOT, "presentation.pdf")
pdf.output(out)
print("сохранено:", out, os.path.getsize(out), "bytes")
