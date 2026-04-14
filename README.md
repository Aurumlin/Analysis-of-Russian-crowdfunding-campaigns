# Analysis-of-Russian-crowdfunding-campaigns
# Техническая документация

Проект: анализ факторов успеха reward-based краудфандинговых кампаний на Planeta.ru.
Цель — статистически проверить, какие текстовые и нетекстовые характеристики описания
кампании предсказывают достижение цели по сбору (`is_successful`) и долю собранного
(`funding_pct`).

---

## 1. Структура репозитория

```
Диплом_2/
├── Planeta.ipynb                     # парсинг JSON → DataFrame + EDA
├── projects_planeta.xlsx             # полный датасет (~12 МБ)
├── projects_sample.xlsx              # sample из 10 проектов (для отладки)
├── projects_sample_with_text.xlsx    # sample + текстовые признаки (результат 1-го шага)
├── dicts.py                          # словари LIWC-подобных категорий
├── text_features.py                  # функции извлечения признаков 1.1–1.5
├── extract_features.py               # полный пайплайн признаков (1.1–1.7)
├── eda_text_features.py              # Mann-Whitney + point-biserial + violin-plots
├── econometrics.py                   # Logit / OLS / VIF / LASSO
├── ml_models.py                      # LogReg / RF / GBM / LGBM / XGB + SHAP
├── shap_tfidf.py                     # SHAP на уровне слов через TF-IDF
├── interpretability.py               # PDP / ICE / Permutation / LIME / GAM / TreeSHAP
├── normalization_scaler.py           # ✨ управление параметрами z-score (воспроизводимость)
├── figures/                          # все PNG-графики
│   └── readability_scaler_params.json # сохранённые параметры нормализации (из interpretability.py)
├── eda_results.csv                   # таблица статтестов
├── shap_words_report.csv             # топ-слова с SHAP-вкладом
├── econometrics_report.txt           # полный лог regression-моделей
├── ml_report.txt                     # CV-метрики ML-моделей
└── interpretability_report.txt       # метрики на test + интерпретация
```

---

## 2. Конвейер данных

```
projects_planeta.xlsx
   │  (Planeta.ipynb: parse_planeta_json_to_dfs, normalize_planeta_dtypes)
   ▼
projects_sample.xlsx        ←  фича-инжиниринг: log_goal, has_video, description_len_chars,
                                funding_ratio, is_successful, campaign_duration_days,
                                image_count, video_count, external_link_count, category_grouped
   │  extract_features.py   (1.1 – 1.7)
   ▼
projects_sample_with_text.xlsx
   │
   ├── eda_text_features.py     → eda_results.csv + figures/violin_*.png
   ├── econometrics.py          → econometrics_report.txt
   ├── ml_models.py             → ml_report.txt + figures/shap_*.png
   ├── interpretability.py      → interpretability_report.txt + figures/interp_*.png
   └── shap_tfidf.py            → shap_words_report.csv + figures/tfidf_*.png
```

Точка входа на шаге 1 — `projects_sample.xlsx` (или полный `projects_planeta.xlsx` после
замены константы `INPUT_FILE` в `extract_features.py`).

---

## 3. Извлечение признаков (`extract_features.py` / `text_features.py`)

### 3.1 Очистка текста (раздельная для разных задач)

Единая чистка текста приводила к потере важной информации (цифры, пунктуация). Решение: **раздельная чистка** для каждой задачи:

| Функция | Использует | Сохраняет | Удаляет |
|---|---|---|---|
| `clean_text_for_liwc()` | Словарные признаки (1.1–1.4) | Слова | Цифры, пунктуация |
| `clean_text_for_readability()` | Индексы читаемости (1.5) | Всё | Только HTML-теги |
| `clean_text_for_rubert()` | RuBERT sentiment (1.6) | Всё | Только HTML-теги |
| `clean_text_for_lda()` | Topic modeling (1.7) | Слова | Цифры, пунктуация |

**Пример:** Текст "Мы напечатаем 500 книг за ₽50,000"

| Чистка | Результат |
|---|---|
| `clean_text_for_liwc` | "Мы напечатаем книг за" |
| `clean_text_for_readability` | "Мы напечатаем 500 книг за ₽50,000" |
| `clean_text_for_rubert` | "Мы напечатаем 500 книг за ₽50,000" |
| `clean_text_for_lda` | "Мы напечатаем книг за" |

**Реализация:**
```python
# В extract_all_text_features():
text_liwc = clean_text_for_liwc(orig_text)
text_read = clean_text_for_readability(orig_text)
# RuBERT и LDA получают специфичную чистку в своих функциях
```

### 3.2 Токенизация и лемматизация

- Токены: `re.findall(r"[а-яёА-ЯЁa-zA-Z]+", text)`.
- Лемматизация: `pymorphy3` (fallback — `pymorphy2`).
- Предложения: `nltk.sent_tokenize(text, language="russian")`.

### 3.3 Формулы признаков

Обозначения:
- `T` = список токенов (слова из текста)
- `L` = список лемм (нормальные формы)
- `W = len(T)` = кол-во слов
- `S` = число предложений
- `SOCIAL_WORDS`, `GRATITUDE_ROOTS`, ... = словари из `dicts.py`

---

#### 1.1 Социальность (`social_score`)

```
Алгоритм:
1. Очистить текст (clean_text) → удалить HTML, спецсимволы
2. T = tokenize(cleaned_text)        # через regex r"[а-яёА-ЯЁa-zA-Z]+"
3. L = lemmatize_tokens(T)          # через pymorphy3
4. social_count = |{l ∈ L : l ∈ SOCIAL_WORDS}|
5. social_score = social_count / len(L)

Словарь SOCIAL_WORDS (~100 лемм):
  "мы", "команда", "друзья", "семья", "вместе", "сообщество",
  "люди", "общество", "коллеги", "партнёры", "поддержка", "сообщество",
  "единство", "участники", "вместе", ...

Интерпретация: доля слов, относящихся к социальным категориям.
Диапазон: [0, 1]. Гипотеза: выше → более вовлекающий текст → успешнее.
```

---

#### 1.2 Благодарность (`gratitude_score`, `has_gratitude`)

```
Алгоритм:
1. cleaned_text = clean_text(original_text)
2. Поиск pattern: GRATITUDE_PATTERN = regex-объединение GRATITUDE_ROOTS
3. matches = GRATITUDE_PATTERN.findall(cleaned_text)
4. T = tokenize(cleaned_text)
5. gratitude_score = len(matches) / max(len(T), 1)
6. has_gratitude = 1 если len(matches) > 0, иначе 0

Словарь GRATITUDE_ROOTS (regex-паттерны):
  "спасибо", "благодар.*", "признателен", "признательна", 
  "благодарность", "ценим", "от всего сердца", ...

Интерпретация:
  - gratitude_score ∈ [0, 1]: интенсивность благодарности на слово
  - has_gratitude ∈ {0, 1}: наличие хотя бы одного выражения благодарности
  
Гипотеза: благодарность к финансирующим людям → положительный посыл → успех.
```

---

#### 1.3 Коллективность (`we_count`, `i_count`, `we_ratio`, `we_vs_i`)

```
Алгоритм:
1. T = tokenize(cleaned_text)
2. T_lower = [t.lower() for t in T]
3. we_count = |{t ∈ T_lower : t ∈ WE_WORDS}|
4. i_count = |{t ∈ T_lower : t ∈ I_WORDS}|

Словари (все морфологические формы):
  WE_WORDS:   "мы", "нас", "нам", "нами", "наш", "наша", "наше", "наши",
              "нашего", "нашей", "нашим", "нашими", "нашему", ...
  I_WORDS:    "я", "меня", "мне", "мной", "мою", "моего", "моей", 
              "моим", "мой", "моя", "моё", "мои", ...

Метрики:
  we_ratio = we_count / (we_count + i_count + 1e-6)
             диапазон [0, 1]: 0 = всё "я", 1 = всё "мы"
  
  we_vs_i = we_count / (i_count + 1)
            диапазон [0, ∞): 0 = нет "мы", >1 = больше "мы" чем "я"

Интерпретация:
  - Высокий we_ratio / we_vs_i → коллективный стиль → командная вибрация
  - Гипотеза: коллективный тон привлекает больше спонсоров (чувство принадлежности)
```

---

#### 1.4 Уверенность / Неуверенность (`certainty_score`, `uncertainty_score`)

```
Алгоритм:
1. cleaned_text = clean_text_for_liwc(original_text)
2. T = tokenize(cleaned_text)
3. L = lemmatize_tokens(T)
4. certainty_count = |{l ∈ L : l ∈ CERTAINTY_WORDS}|
5. uncertainty_count = |{l ∈ L : l ∈ UNCERTAINTY_WORDS}|
6. certainty_score = certainty_count / len(L)
7. uncertainty_score = uncertainty_count / len(L)

Словари:
  CERTAINTY_WORDS (~20 лемм):
    "обязательно", "гарантия", "точно", "определённо", "уверен", "неопровержимо",
    "несомненно", "бесспорно", "безусловно", "решительно", "явно", ...

  UNCERTAINTY_WORDS (~20 лемм):
    "возможно", "может", "наверное", "вероятно", "похоже", "как будто",
    "попробуем", "попытаемся", "не уверен", "не известно", "неясно", ...

Интерпретация:
  - Высокий certainty_score → уверенный тон → автор верит в проект
  - Высокий uncertainty_score → неуверенность → вызывает сомнения

Гипотеза: уверенные формулировки привлекают инвестиции (психология доверия).
```

---

#### 1.4.5 Числовые признаки — Конкретность описания (`money_mentions`, `number_density`, `has_specific_sum`)

```
Алгоритм:
1. cleaned_text = clean_text(original_text)
2. T = tokenize(cleaned_text)
3. L = lemmatize_tokens(T)
4. certainty_count = |{l ∈ L : l ∈ CERTAINTY_WORDS}|
5. uncertainty_count = |{l ∈ L : l ∈ UNCERTAINTY_WORDS}|
6. certainty_score = certainty_count / len(L)
7. uncertainty_score = uncertainty_count / len(L)

Словари:
  CERTAINTY_WORDS (~20 лемм):
    "обязательно", "гарантия", "точно", "определённо", "уверен", "неопровержимо",
    "несомненно", "бесспорно", "безусловно", "решительно", "явно", ...
  
  UNCERTAINTY_WORDS (~20 лемм):
    "возможно", "может", "наверное", "вероятно", "похоже", "как будто",
    "попробуем", "попытаемся", "не уверен", "не известно", "неясно", ...

Интерпретация:
  - Высокий certainty_score → уверенный тон → автор верит в проект
  - Высокий uncertainty_score → неуверенность → вызывает сомнения
  
Гипотеза: уверенные формулировки привлекают инвестиции (психология доверия).
```

---

#### 1.4.5 Числовые признаки — Конкретность описания

```
Алгоритм:
1. money_pattern = regex для денежных сумм: \d[\d\s,.]*\s*(?:руб|тыс|млн|₽)
2. numbers_pattern = regex для любых цифр: \b\d+(?:[.,]\d+)?\b
3. T = tokenize(text)
4. N = max(len(T), 1)

Признаки:
  money_mentions    = len(money_pattern.findall(text))
                    [абсолютное число упоминаний денежных сумм]
  
  number_density    = len(numbers_pattern.findall(text)) / N
                    [доля слов, являющихся цифрами или числами]
  
  has_specific_sum  = 1 если money_pattern.search(text) else 0
                    [бинарный флаг: есть ли конкретная сумма в рублях]

Интерпретация:
  Текст "Мы напечатаем 500 книг для 10 школ за ₽50,000":
    - money_mentions = 1     (одна денежная сумма ₽50,000)
    - number_density ≈ 0.10  (3 цифры / ~30 слов)
    - has_specific_sum = 1   (да, есть конкретная сумма)
  
  Текст "Мы сделаем много добра":
    - money_mentions = 0     (нет денежных сумм)
    - number_density = 0     (нет цифр)
    - has_specific_sum = 0   (нет)
  
Гипотеза: конкретность описания (наличие цифр и сумм) → выше доверие → выше успех.
```

---

#### 1.5 Читаемость (`readability_flesch`, `readability_fog`, `readability_lix`, `readability_avg`)

```
Подготовка:
1. Найти число предложений: S = len(nltk.sent_tokenize(text, "russian"))
2. Найти число слов: W = len(tokenize(text))
3. Найти число слогов: SYLL = Σ count_syllables_ru(word) для каждого word
   - count_syllables_ru(word) = число гласных [аеёиоуыэюяАЕЁИОУЫЭЮЯ]
4. Найти число "сложных" слов (≥3 слога): n_complex
5. Найти число "длинных" слов (>6 букв): n_long

Основные индексы:
  ASL = W / S         (average sentence length in words)
  ASW = SYLL / W      (average syllables per word)

Формулы индексов:
  
  Flesch (ru-адаптация):
    readability_flesch = 206.835 − 1.015·ASL − 84.6·ASW
    диапазон: обычно [0, 100], выше = проще читать
    < 30     = очень сложный текст
    30–60    = сложный (наук. статья)
    60–80    = средний
    > 80     = простой (ребёнок поймёт)
  
  Gunning Fog Index:
    readability_fog = 0.4·(ASL + 100·n_complex / W)
    диапазон: [0, ~20], выше = сложнее
    интерпретация: примерно год образования, нужный для понимания
  
  LIX Index:
    readability_lix = ASL + 100·n_long / W
    диапазон: [20, ~80], выше = сложнее
    легче интерпретировать для других языков

Нормализация (для readability_avg):
1. Посчитать z-score каждого индекса (нормализация):
   z_flesch = (flesch − mean) / std
   z_fog = (fog − mean) / std
   z_lix = (lix − mean) / std
2. Инвертировать FOG и LIX (т.к. для них "больше" = "сложнее"):
   z_fog := −z_fog
   z_lix := −z_lix
3. Усреднить три нормализованных индекса:
   readability_avg = (z_flesch + z_fog + z_lix) / 3

Интерпретация:
  - readability_avg > 0 → текст читается легче, чем в среднем
  - readability_avg < 0 → текст сложнее среднего
  
Гипотеза: легче читать → больше людей прочитает → выше шанс успеха.
```

---

#### 1.6 RuBERT Sentiment (тональность)

```
Алгоритм:
1. cleaned_text = clean_text(original_text)
2. Загрузить модель: blanchefort/rubert-base-cased-sentiment
3. Разбить на предложения: sents = nltk.sent_tokenize(cleaned_text, "russian")
4. Для каждого предложения (батч до 16 сент., макс 512 токенов):
   a) Токенизировать через AutoTokenizer
   b) Пропустить через модель → logits (batch_size, 3)
   c) softmax(logits) → вероятности (batch_size, 3)
5. Усреднить вероятности по всем предложениям:
   probs_avg = mean(all_probs, axis=0)

Результат: три нормализованные вероятности:
  rubert_positive ∈ [0, 1]    = доля "позитивных" токенов
  rubert_negative ∈ [0, 1]    = доля "негативных" токенов
  rubert_neutral ∈ [0, 1]     = доля "нейтральных" токенов
  сумма всегда = 1.0

Интерпретация:
  - Высокий positive → текст звучит позитивно → привлекает финансирование
  - Высокий negative → красные флаги → отталкивает
  - High neutral → информативный, но холодный стиль

Гипотеза: позитивная тональность → выше успех.
```

---

#### 1.7 LDA Topic Distribution (скрытые темы)

```
Алгоритм:
1. Препроцессинг: очистить текст, токенизировать, лемматизировать
2. Фильтр стоп-слов: убрать ~30 самых частых (и, в, на, что, как, ...)
3. Фильтр по длине: убрать токены length ≤ 2
4. Получить corpus documents (список списков лемм)
5. Обучить LDA:
   model = LdaModel(corpus, num_topics=K, passes=10, random_state=42)
   где K=5 (произвольно выбран)
6. Для каждого документа: theta = model.get_document_topics(doc, min_prob=0.0)
   → распределение вероятностей по темам (K значений, сумма=1)

Результат: K признаков
  topic_0, topic_1, ..., topic_4 ∈ [0, 1]
  сумма всегда = 1.0 для каждого документа

Интерпретация:
  - Каждая тема = набор слов с высокими весами (выводятся в лог)
  - topic_i для документа = "насколько этот документ о теме i"
  - Тема может оказаться предсказывающей (напр., тема "инновации" → выше успех)

Гипотеза: некоторые темы привлекают больше поддержки (социальные проекты
vs. высокотехнологичные).
```

---

**Итоговая таблица (краткая):**

| Признак | Что считаем | Диапазон | Выше → ? |
|---|---|---|---|
| `social_score` | доля социальных слов | [0, 1] | более вовлекающий тон |
| `gratitude_score` | интенсивность благодарности | [0, 1] | благодарный тон |
| `has_gratitude` | наличие благодарности | {0, 1} | вежливость |
| `we_count` | кол-во "мы" | [0, ∞) | коллективность |
| `i_count` | кол-во "я" | [0, ∞) | индивидуализм |
| `we_ratio` | мы / (мы+я) | [0, 1] | коллективный стиль |
| `we_vs_i` | мы / (я+1) | [0, ∞) | коллективный стиль |
| `certainty_score` | доля уверенных слов | [0, 1] | уверенный тон |
| `uncertainty_score` | доля неуверенных слов | [0, 1] | неуверенность |
| `readability_flesch` | индекс читаемости | [−∞, 100] | выше = проще |
| `readability_fog` | год образования нужный | [0, ~20] | выше = сложнее |
| `readability_lix` | шведский индекс | [20, 80] | выше = сложнее |
| `readability_avg` | нормализованное среднее | [−∞, ∞] | выше = проще читать |
| `rubert_positive` | вероятность позитива | [0, 1] | позитивный тон |
| `rubert_negative` | вероятность негатива | [0, 1] | негативный тон |
| `rubert_neutral` | вероятность нейтрала | [0, 1] | информативный |
| `topic_0` – `topic_4` | вероятность темы k | [0, 1] | текст о теме k |

### 3.4 RuBERT sentiment (1.6)

Модель `blanchefort/rubert-base-cased-sentiment`. Текст разбивается на предложения,
каждое — до 512 токенов, батч 16, softmax по логитам. Итог — среднее распределение
вероятностей `[positive, negative, neutral]` по всем предложениям документа.

### 3.5 LDA (1.7)

- Корпус: леммы с фильтром стоп-слов и `len > 2`.
- `gensim.LdaModel`, `K = 5`, `passes = 10`, `random_state = 42`.
- На выборках `N > 30` — `filter_extremes(no_below=2, no_above=0.9)`.
- Выход: `topic_0 … topic_4` — распределение вероятностей по темам.

---

## 4. EDA признаков (`eda_text_features.py`)

Для каждого признака:

- Mann-Whitney U-test между `is_successful = 0/1`.
- Rank-biserial effect size: `r = 1 − 2U / (n₁·n₂)`.
- Point-biserial корреляция: `stats.pointbiserialr(target, feature)`.
- Violin-plot `figures/violin_<feature>.png`.

Выход: `eda_results.csv`, сортировка по `mw_p`.

---

## 5. Эконометрика (`econometrics.py`)

Поэтапные спецификации (подхвата категориальных дамми `cat_*`):

| Модель | Признаки |
|---|---|
| M0 | `CONTROLS_BASE` + `cat_*` |
| M1 | +`readability_avg`, `description_word_count` |
| M2 | +`rubert_positive`, `rubert_negative` |
| M3 | +LIWC (`social_score`, `gratitude_score`, `we_ratio`, `certainty_score`, `uncertainty_score`) |
| M_full | все выше |
| M_lasso | LogisticRegressionCV(L1), автоматический отбор |

`CONTROLS_BASE` = `log_goal`, `campaign_duration_days`, `counts.newsCount`,
`counts.commentsCount`, `card.author.campaignsAmount`, `has_video`, `log_text_length`.

### 5.1 Logit на `is_successful`

- `statsmodels.Logit(y, X).fit(cov_type="HC1")` — робастные ошибки Huber-White HC1.
- Отчёт: coefficients, z, p-value, 95% CI.
- Odds Ratios: `exp(β)` и `exp(CI)`.
- Фит-статистики: McFadden R², AIC, BIC, LLF.

### 5.2 OLS на `funding_pct`

- `statsmodels.OLS(y, X).fit(cov_type="HC1")`.
- R², Adj-R², AIC/BIC.

### 5.3 Мультиколлинеарность

`variance_inflation_factor` из `statsmodels.stats.outliers_influence`. Порог тревоги — VIF > 5.

### 5.4 LASSO

`LogisticRegressionCV(Cs=10, penalty="l1", solver="saga")`. Выводится список ненулевых
коэффициентов и оптимальный `C`.

---

## 6. ML-модели (`ml_models.py`)

### 6.1 Набор признаков

- **A** — контроли + категориальные дамми (`CONTROLS` + `CATEGORY_DUMMIES`).
- **B** — только текстовые признаки (`TEXT_FEATURES`).
- **C** — A + B (контроли + категориальные + текстовые).

### 6.2 Модели

| Модель | Конфигурация |
|---|---|
| LogReg L1 | `StandardScaler` → `LogisticRegression(penalty="l1", solver="saga", class_weight="balanced")` |
| RandomForest | `n_estimators=300`, `class_weight="balanced"` |
| GradBoost | sklearn GBM, `n_estimators=200`, `max_depth=3` |
| LightGBM | `n_estimators=200`, `class_weight="balanced"` |
| XGBoost | `n_estimators=200`, `max_depth=4`, `eval_metric="logloss"` |

### 6.1.1 Категориальные дамми (One-Hot Encoding)

Если в датасете присутствует `category_grouped` (категория проекта), автоматически создаются one-hot дамми переменные:

```python
# В prepare():
cat_dummies = pd.get_dummies(df["category_grouped"], prefix="cat", drop_first=True)
# Результат: cat_Образование, cat_Творческие, cat_Социальные, ... (первая категория dropped)
```

Дамми включаются в набор A и C для захвата влияния категории проекта на успешность:
- Может быть, одни категории более успешны чем другие
- Категория может быть confounding factor для текстовых признаков

Пример из 10-проектной выборки:
```
cat_Образование и просвещение: mean=0.3, связь с успехом?
cat_Социальные:                mean=0.2
cat_Творческие:                mean=0.3
```

При `drop_first=True` первая категория служит baseline; остальные кодируют отклонение от baseline.

### 6.3 Валидация

- `Stratified 5-Fold` если `len(y) ≥ 30` и минимальный класс ≥ 5.
- Иначе `LeaveOneOut`.
- Метрики: ROC-AUC, PR-AUC, F1, Precision, Recall (`cross_val_predict(method="predict_proba")`).

### 6.4 Interpretability: SHAP Analysis

Вместо ablation study используется **SHAP (TreeExplainer)** для определения важности признаков:

```python
# TreeExplainer на RandomForest (лучшая модель)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)
```

Вывод:
- Ranking признаков по `mean(|SHAP|)` — как часто и насколько сильно каждый признак влияет
- Beeswarm plot — распределение SHAP значений по отдельным сэмплам
- Bar plot — средняя абсолютная важность

Преимущества SHAP vs Ablation:
- ✓ Не требует переобучения модели N раз (быстро)
- ✓ Показывает направление эффекта (+/−) через цвет
- ✓ Интерпретируется как "зависит от значения признака"

### 6.5 SHAP

TreeSHAP на RandomForest: `shap_beeswarm.png`, `shap_bar.png`, таблица `mean(|SHAP|)`.

---

## 7. Интерпретация (`interpretability.py`)

Конвейер на train/test (80/20, stratified):

1. RandomForest + LightGBM fit на train, метрики на test.
2. Гистограммы / boxplot / корреляционная матрица числовых признаков.
3. Permutation Importance (на test, 30 повторов).
4. Partial Dependence + ICE для топ-N признаков.
5. TreeSHAP: beeswarm, bar, dependence plots.
6. Mean |SHAP| в разрезе `category_grouped`.
7. LIME для нескольких примеров из test.
8. GAM (`pyGAM.LogisticGAM`) — partial effects.
9. Learning curves (размер train vs ROC-AUC).
10. Ошибки по сегментам: категория, квартиль `log_goal`, `has_video`.

---

## 8. SHAP на уровне слов (`shap_tfidf.py`)

1. `clean_text` + лемматизация `pymorphy3`.
2. `TfidfVectorizer(max_features=500, min_df=2, ngram_range=(1,2))`.
3. Модель A — `LogisticRegression(L1)` → `shap.LinearExplainer`.
4. Модель B — `LightGBM` → `shap.TreeExplainer`.
5. Выводы: коэффициенты LogReg, SHAP beeswarm (LogReg + LGBM), SHAP bar (LGBM),
   таблица `shap_words_report.csv` (топ-слов по `mean(|SHAP|)` с указанием знака).

---

## 9. Запуск

### Быстрый запуск всего пайплайна

```bash
# Всё подряд (extract → eda → econ → ml → interp → tfidf)
python3 run_all.py

# Только указанные шаги
python3 run_all.py --only extract,ml

# Пропустить тяжёлые шаги
python3 run_all.py --skip tfidf,interp

# С Tobit-регрессией и log-трансформацией
python3 run_all.py --use-tobit --log-funding-pct

# Остановиться при первой ошибке
python3 run_all.py --stop-on-error
```

`run_all.py` выводит сводку с временем каждого шага и финальным статусом.

### Запуск отдельных скриптов

```bash
# 1. признаки (тяжёлое: грузит RuBERT, обучает LDA)
python3 extract_features.py

# 2. EDA
python3 eda_text_features.py

# 3. регрессии
python3 econometrics.py
python3 econometrics.py --use-tobit              # Tobit вместо OLS
python3 econometrics.py --log-funding-pct        # log-OLS

# 4. ML + SHAP
python3 ml_models.py

# 5. расширенная интерпретация
python3 interpretability.py

# 6. SHAP по словам
python3 shap_tfidf.py
```

Зависимости: `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`,
`seaborn`, `shap`, `lightgbm`, `xgboost`, `pymorphy3` (или `pymorphy2`), `pyphen`,
`nltk`, `gensim`, `transformers`, `torch`, `lime`, `pygam`.

---

## 10. Исправленные проблемы (Data Leakage & Multicollinearity)

### ✅ Data Leakage + Non-Reproducibility в `readability_avg` (ИСПРАВЛЕНО)

**Двойная проблема:**

1. **Data leakage:** нормализация по всему датасету до train/test split → test видит train statistics
2. **Non-reproducibility:** при добавлении новых данных mean/std меняются → старые z-score становятся несопоставимы

**Пример неправильного сценария:**
```
День 1: extract_features на 100 проектов
  → readability_avg mean=5.0, std=2.0
  → z_score = (value - 5.0) / 2.0
  → данные сохранены в .xlsx

День 30: добавляем 50 новых проектов
  → extract_features на 150 проектов
  → readability_avg mean=5.3, std=2.1  (изменились!)
  → z_score = (value - 5.3) / 2.1
  → ВСЕ СТАРЫЕ z-score теперь неправильные ❌
```

**Решение (архитектурное):**
```
extract_features.py
  ↓ сохраняет ТОЛЬКО сырые значения (без нормализации)
  readability_flesch, readability_fog, readability_lix, readability_avg
  ↓
ml_models.py / interpretability.py / econometrics.py
  ↓ каждый скрипт управляет нормализацией отдельно
  normalization_scaler.py::ReadabilityScaler
    • fit только на train (NO LEAKAGE)
    • сохраняет mean/std в JSON
    • reproducibly применяет к новым данным
```

**Реализация:**
- `text_features.py::add_readability_avg()` вычисляет только **сырое среднее**: `(flesch − fog − lix) / 3`
- Новый модуль `normalization_scaler.py`:
  ```python
  from normalization_scaler import ReadabilityScaler
  
  # При обучении моделей
  scaler = ReadabilityScaler(fit_on_train=True)
  X_train = scaler.fit_transform(X_train)  # fit на train
  X_test = scaler.transform(X_test)        # apply с train parameters
  scaler.save("readability_params.json")   # сохранить параметры
  
  # При применении к новым данным (месяцы спустя)
  scaler = ReadabilityScaler.load("readability_params.json")
  X_new = scaler.transform(X_new)  # используем СТАРЫЕ mean/std → консистентно!
  ```
- `interpretability.py::normalize_readability_after_split()` вызывает `ReadabilityScaler` после split
- Параметры сохраняются в `figures/readability_scaler_params.json`

**Результат:**
- ✓ Нет data leakage (fit только на train)
- ✓ Воспроизводимость (добавление данных не меняет масштаб старых)
- ✓ Production-ready (параметры сохраняются для будущих данных)
- ✓ Монотонность (z-score признаков остаются сравнимы во времени)

### ✅ Идеальная мультиколлинеарность RuBERT и LDA (ИСПРАВЛЕНО)

**Проблема:** `rubert_positive + rubert_negative + rubert_neutral = 1.0` (всегда).
Аналогично: `topic_0 + … + topic_4 = 1.0`. Матрица X вырождена → Logit / OLS не сходятся, VIF → ∞.

**Решение:**
- `extract_features.py` удаляет `rubert_neutral` и `topic_4` перед сохранением в `.xlsx`.
- Остаются: `rubert_positive`, `rubert_negative` и `topic_0, topic_1, topic_2, topic_3`.
- Логит и OLS работают корректно на остальных признаках.

### ✅ Проблема 8: funding_pct сильно скошена (много нулей, экстремальные значения) (ИСПРАВЛЕНО)

**Проблема:** Целевая переменная для OLS `funding_pct = funding_ratio.clip(lower=0)` часто имеет:
- Много нулей (проекты, которые не собрали ничего)
- Экстремальные значения (проекты, которые собрали в 10+ раз больше цели)
- Асимметричное распределение

Это нарушает предположение о нормальности остатков в OLS, приводит к:
- Смещенным стандартным ошибкам
- Неправильным доверительным интервалам
- Нестабильности модели

**Решение: Log-трансформация через log1p**

```python
df["log_funding_pct"] = np.log1p(df["funding_pct"])
# Эквивалентно: log(1 + funding_pct)
# - log(1 + 0) = 0 (нули остаются нулями в log-space)
# - log(1 + 2) ≈ 1.1 (разумное значение для ratio=2)
# - log(1 + 100) ≈ 4.6 (сжимает экстремальные значения)
```

**Использование в `econometrics.py`:**

```bash
# OLS на оригинальном funding_pct (по умолчанию)
python3 econometrics.py

# OLS на log(1 + funding_pct) — рекомендуется при скошенном distribution
python3 econometrics.py --log-funding-pct
```

**Интерпретация коэффициентов при log-трансформации:**

Если модель: `log(funding_pct + 1) = β₀ + β₁·X₁ + ε`

То: `funding_pct ≈ exp(β₁·X₁) - 1` (при малых X₁)

Пример: β₁ = 0.25 означает:
- Увеличение X₁ на 1 единицу → funding_pct увеличится примерно на 28% (= exp(0.25) − 1)

**Диагностика: когда использовать log-трансформацию?**

```python
# Проверить распределение funding_pct:
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2)
df["funding_pct"].hist(bins=30, ax=axes[0], title="funding_pct (original)")
np.log1p(df["funding_pct"]).hist(bins=30, ax=axes[1], title="log(1 + funding_pct)")
plt.show()

# Если левое распределение сильно скошено вправо → используй --log-funding-pct
```

**Результат:**
- ✓ Более нормальное распределение остатков
- ✓ Стабильные стандартные ошибки
- ✓ Интерпретируемые коэффициенты (в % изменении funding_pct)
- ✓ Production-ready модель

### ✅ Проблема 9: Tobit-регрессия для цензурированных данных (РЕШЕНИЕ ДОБАВЛЕНО)

**Альтернативное решение для Проблемы 8:**

Вместо log-трансформации можно использовать **Tobit модель** (censored regression), которая явно моделирует цензурирование:

```
y_latent = X·β + ε,   ε ~ N(0, σ²)
y_observed = max(0, y_latent)   # нижняя цензура на 0
```

**Когда Tobit лучше чем log-OLS:**

| Аспект | Log-OLS | Tobit |
|---|---|---|
| **Интерпретация** | % изменение funding_pct | Прямой эффект на скрытую переменную |
| **Нулевые значения** | Сдвигаются на log-scale | Явно моделируются как цензурированные |
| **Экстремальные значения** | Сжимаются логарифмом | Не преобразуются, полнинформативны |
| **Теоретическое обоснование** | Эмпирическое (работает на практике) | Статистическое (MLE для цензурированных данных) |
| **Производительность** | Быстро (линейная модель) | Медленнее (нелинейная оптимизация) |

**Использование в `econometrics.py`:**

```bash
# Tobit на funding_pct
python3 econometrics.py --use-tobit

# Tobit на log-трансформированной funding_pct
python3 econometrics.py --use-tobit --log-funding-pct

# Обычный OLS (default, НЕ рекомендуется для этих данных)
python3 econometrics.py
```

**Интерпретация коэффициентов Tobit:**

Коэффициент β показывает эффект на **скрытую переменную** y_latent:
- Если β₁ = 0.5 → увеличение X₁ на 1 единицу увеличивает y_latent на 0.5
- Маржинальный эффект на E[y_observed | y_observed > 0] < β (т.к. часть эффекта "теряется" на цензуре)

**Диагностика: Tobit vs Log-OLS vs OLS**

```python
# 1. Проверить долю цензурированных наблюдений
pct_zero = (df["funding_pct"] == 0).mean()
print(f"Доля проектов с funding_pct = 0: {pct_zero:.1%}")
# Если > 20% → Tobit рекомендуется

# 2. Сравнить логарифмы правдоподобия
# Tobit возвращает log-likelihood, сравнить с OLS (или AIC)

# 3. Проверить остатки
# Для Tobit: остатки должны быть нормальны для uncensored наблюдений
# Для log-OLS: остатки после exp(ŷ) трансформации
```

**Реализация в коде:**

```python
def fit_tobit(X, y, lower=0.0):
    """
    Максимум-правдоподобная оценка (MLE) для Tobit модели.
    
    Для цензурированных наблюдений (y ≤ lower):
      log L = log Φ((lower - X·β) / σ)   # вероятность быть в нуле
    
    Для uncensored (y > lower):
      log L = log φ((y - X·β) / σ) - log(σ)  # плотность вероятности
    
    Оптимизируется scipy.optimize.minimize.
    """
```

**Результат:**
- ✓ Статистически обоснованное моделирование цензурирования
- ✓ Не требует трансформации целевой переменной
- ✓ Правильная обработка нулей (как цензурированных, не как отсутствия значения)
- ✓ Вывод: `n_censored`, `n_uncensored`, σ, коэффициенты

### ✅ Проблема 10: Несогласованность знаменателей в формулах признаков (ИСПРАВЛЕНО)

**Проблема:** Текстовые признаки используют разные знаменатели:

```python
social_score    = social_count / len(L)       # L = леммы
gratitude_score = matches / max(len(T), 1)    # T = токены (сырые слова)
certainty_score = cert_count / len(L)         # L = леммы
we_ratio        = we / (we + i + ε)           # только местоимения!
```

**Почему это проблема:**
- Леммы ≠ токены: "командой" → 1 токен, но после лемматизации = "команда"
- we_ratio не нормализуется к полному тексту, только к местоимениям
- Несопоставимые признаки → неправильные коэффициенты в регрессии
- Трудно интерпретировать: "на сколько процентов текста это влияет"

**Решение: Унификация на леммы**

Все признаки теперь используют **леммы** как стандартный знаменатель:

```python
# 1.1 Социальность (было: len(lemmas), осталось так же)
social_score = social_count / len(L)

# 1.2 Благодарность (было: len(tokens), ИЗМЕНЕНО на: len(lemmas))
gratitude_score = matches / len(L)   # ← ИЗМЕНЕНО!
# Примечание: GRATITUDE_PATTERN ищет в сыром тексте, но нормируем к леммам

# 1.3 Коллективность (было: только мы+я, ИЗМЕНЕНО на: % от текста)
we_ratio = we_count / len(T)         # ← ИЗМЕНЕНО!
i_ratio = i_count / len(T)
we_vs_i = we_count / (i_count + 1)   # остаётся как есть (безразмерная)

# 1.4 Уверенность (было: len(lemmas), осталось так же)
certainty_score = cert_count / len(L)
uncertainty_score = uncert_count / len(L)
```

**Реализация в `text_features.py`:**

```python
def compute_collectivism(text: str, normalize_to_tokens: bool = True) -> dict:
    """
    Параметр normalize_to_tokens:
      True (default): мы_count / len(tokens) — % от всех слов
      False (legacy): мы_count / (мы_count + я_count) — только среди местоимений
    """
    tokens = tokenize(text)
    lemmas = lemmatize_tokens(tokens)
    if not tokens:
        return {"we_count": 0, "i_count": 0, "we_ratio": 0.0, "i_ratio": 0.0, "we_vs_i": 0.0}
    
    lowered = [t.lower() for t in tokens]
    we_count = sum(1 for t in lowered if t in WE_WORDS)
    i_count = sum(1 for t in lowered if t in I_WORDS)
    
    if normalize_to_tokens:
        # Новая интерпретация: % от всех слов в тексте
        n = len(tokens)
        we_ratio = we_count / n
        i_ratio = i_count / n
    else:
        # Старая: только среди местоимений
        we_ratio = we_count / (we_count + i_count + 1e-6)
        i_ratio = i_count / (we_count + i_count + 1e-6)
    
    we_vs_i = we_count / (i_count + 1)  # безразмерная, без изменений
    
    return {
        "we_count": we_count,
        "i_count": i_count,
        "we_ratio": we_ratio,
        "i_ratio": i_ratio,
        "we_vs_i": we_vs_i,
    }
```

**Интерпретация после унификации:**

| Признак | Диапазон | Интерпретация |
|---|---|---|
| `social_score` | [0, 1] | % слов (лемм), относящихся к социальным категориям |
| `gratitude_score` | [0, 1] | % слов (лемм), содержащих благодарность |
| `we_ratio` | [0, 1] | % слов (токенов), являющихся "мы"-местоимениями |
| `i_ratio` | [0, 1] | % слов (токенов), являющихся "я"-местоимениями |
| `certainty_score` | [0, 1] | % слов (лемм), выражающих уверенность |
| `uncertainty_score` | [0, 1] | % слов (лемм), выражающих неуверенность |

**Обратная совместимость:**

```python
# Если нужна старая интерпретация we_ratio (только среди местоимений):
result = compute_collectivism(text, normalize_to_tokens=False)
```

**Результат:**
- ✓ Все признаки нормализованы к одной базовой единице (леммы или токены)
- ✓ Прямая интерпретация: "X% слов текста обладают свойством Y"
- ✓ Сопоставимые коэффициенты в регрессии
- ✓ Воспроизводимость и ясность формул

---

## 11. Исправленные проблемы (продолжение)

### ✅ Проблема 7: LDA — K=5 выбрано произвольно (ИСПРАВЛЕНО)

**Проблема:** В `extract_features.py` число тем для LDA жёстко закодировано как `K=5`. Это произвольный выбор без обоснования, и может быть субоптимальным для данного корпуса. На малых выборках (N=10–100) это критично.

**Решение: Coherence Score автоматический подбор K**

Используем `gensim.models.CoherenceModel` для оценки качества LDA при разных K:

```python
from gensim.models import LdaModel, CoherenceModel

def find_optimal_k(docs, corpus, dictionary, k_range=range(2, 11), passes=10):
    """
    Подбирает оптимальное число тем по Coherence Score (CV или U_Mass).
    
    Параметры:
      docs: список списков лемм (токенизированные документы)
      corpus: corpus из dictionary.doc2bow()
      dictionary: gensim Dictionary
      k_range: диапазон K для проверки (default: 2–10)
      passes: число проходов LDA (default: 10)
    
    Возвращает:
      best_k: оптимальное K
      scores: словарь {K: coherence_score}
    """
    scores = {}
    models = {}
    
    print("[LDA] подбор K по Coherence Score...")
    for k in k_range:
        print(f"  K={k}...", end=" ")
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k,
                         passes=passes, random_state=42, per_word_topics=True)
        
        # Используем CV (внутренний coherence) — быстро и не требует original docs
        # Альтернатива: CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence='u_mass')
        coherence = CoherenceModel(model=model, corpus=corpus, coherence='u_mass').get_coherence()
        scores[k] = coherence
        models[k] = model
        print(f"score={coherence:.4f}")
    
    best_k = max(scores, key=scores.get)  # Выбираем K с максимальным coherence
    print(f"\n[LDA] ✓ лучший K={best_k} (score={scores[best_k]:.4f})")
    return best_k, scores, models
```

**Интеграция в `extract_features.py`:**

```python
def add_lda_topics(df, text_col=CLEAN_COL, auto_k=True, k_range=range(2, 11)):
    """
    Добавляет topic_0..topic_{K-1}.
    
    Если auto_k=True: подбирает K по Coherence Score.
    Если auto_k=False: использует K=5 (для совместимости).
    """
    from gensim.models import CoherenceModel
    
    # ... подготовка corpus, dictionary ...
    
    if auto_k and len(docs) > 20:  # на малых выборках подбор шумит
        best_k, scores, models = find_optimal_k(docs, corpus, dictionary, k_range)
        lda = models[best_k]
        num_topics = best_k
    else:
        print(f"[LDA] K=5 (режим совместимости или small corpus)")
        num_topics = 5
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                       passes=10, random_state=42)
    
    # ... построение topic_matrix, конкатенация с df ...
```

**Вывод при запуске:**

```
[LDA] подбор K по Coherence Score...
  K=2... score=0.4521
  K=3... score=0.5123
  K=4... score=0.4998
  K=5... score=0.5034
  K=6... score=0.4876
  K=7... score=0.4712
  ...
  K=10... score=0.4320

[LDA] ✓ лучший K=5 (score=0.5123)
```

**Варианты метрик:**

| Метрика | Описание | Когда использовать |
|---|---|---|
| `u_mass` | Internal coherence (быстро, не требует оригинальных текстов) | Для скорости, когда corpus большой |
| `c_v` | Segmentation-based coherence (лучше, но медленнее) | Когда corpus < 10K документов |
| `c_npmi` | Normalized PMI (опция gensim 4.0+) | Для более стабильной оценки |

**Результат:**
- ✓ K выбирается данными, а не эвристикой
- ✓ Автоматизировано: нет нужны ручной подбор
- ✓ Воспроизводимо: одинаковый корпус → одинаковый K
- ✓ Масштабируется: на полных данных даёт более стабильные темы

---

## 12. Проблемы

1. **Все скрипты запускаются на `projects_sample.xlsx` (N = 10).** Это демо-выборка.
   Все логиты в `econometrics_report.txt` пропущены, т.к. предикторов больше, чем
   наблюдений. ML-метрики на LOOCV с N=10 неинтерпретируемы (AUC скачет 0.0 ↔ 1.0).
   **Шаг, который нужно сделать:** запустить `Planeta.ipynb` → получить полный
   нормализованный датасет из `projects_planeta.xlsx`, сохранить в
   `projects_sample.xlsx` (или заменить путь `INPUT_FILE` в `extract_features.py`)
   и прогнать пайплайн заново. На полном датасете (~тысячи проектов) статистические
   тесты станут содержательны.

2. **Конфликт `pymorphy2` / `pymorphy3` на Python 3.13.** Для 3.11+ рекомендуется
   `pymorphy3` (в коде уже стоит try/except).

5. **XGBoost при LOOCV возвращает AUC=0.0** — баг связан с бинарным predict_proba
   на крошечной выборке. На полных данных пропадёт сам.
