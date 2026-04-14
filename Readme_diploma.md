# Словарь данных и карта признаков

## `projects_sample_with_text.xlsx`

Файл `projects_sample_with_text.xlsx` содержит итоговую табличную выборку для текстового анализа и моделирования. В текущей версии в нём 64 столбца.

Важно:

- Колонки `rubert_neutral` и `topic_4` намеренно удаляются перед сохранением файла, чтобы избежать идеальной коллинеарности в регрессиях.
- Часть признаков для моделей создаётся уже внутри скриптов и потому **не хранится как отдельные колонки** в `.xlsx`: `log_goal`, `log_text_length`, `has_video`, `cat_*`, `funding_pct`, `log_funding_pct`.

---

## 1. Описание всех столбцов `projects_sample_with_text.xlsx`

### 1.1 Служебные, идентификационные и исходные поля

| Столбец | Роль | Описание |
| --- | --- | --- |
| `project_key` | идентификатор | Уникальный ключ проекта в собранной таблице |
| `sourceUrl` | источник | URL страницы проекта |
| `card.title` | исходный текст | Заголовок проекта |
| `card.subtitle` | исходный текст | Подзаголовок проекта |
| `card.collectedAmount.value` | финансы | Сумма фактически собранных средств |
| `card.targetAmount.value` | финансы | Целевая сумма сбора |
| `card.daysToFinish` | платформа | Число дней до завершения на момент парсинга карточки |
| `card.startAt` | дата | Дата и время запуска кампании |
| `card.finishAt` | дата | Дата и время завершения кампании |
| `card.region` | метаданные | Регион, указанный в карточке проекта |
| `card.mainCategory.tagName` | категория | Исходное категориальное название проекта на платформе |
| `card.author.id` | автор | ID автора проекта |
| `card.author.campaignsAmount` | автор | Сколько кампаний у автора на платформе |
| `card.links.vk_url` | внешняя ссылка | Ссылка на VK автора/проекта |
| `card.links.telegram_url` | внешняя ссылка | Ссылка на Telegram автора/проекта |
| `card.links.author_site_url` | внешняя ссылка | Ссылка на внешний сайт автора/проекта |
| `description.text` | исходный текст | Основной текст описания проекта |
| `counts.newsCount` | вовлечённость | Количество новостных обновлений проекта |
| `counts.commentsCount` | вовлечённость | Количество комментариев |
| `counts.participantsCount` | вовлечённость | Количество участников / поддержавших |
| `counts.purchasesCount` | вовлечённость | Количество покупок / оформленных поддержек |
| `meta.description` | метаданные | Meta description страницы проекта |
| `rewards.totalRewards` | вознаграждения | Общее число reward tiers в карточке |

### 1.2 Инженерные признаки кампании и метаданных

| Столбец | Роль | Описание |
| --- | --- | --- |
| `image_count` | медиа | Количество изображений, связанных с проектом |
| `video_count` | медиа | Количество видео, связанных с проектом |
| `reward_count` | вознаграждения | Количество наград в нормализованной reward-таблице |
| `external_link_count` | внешние связи | Количество внешних ссылок в проекте |
| `funding_ratio` | целевая метрика | Доля выполнения цели: `collected / target` |
| `is_successful` | целевая метрика | Бинарный флаг успеха: достиг ли проект целевой суммы |
| `campaign_duration_days` | время | Длительность кампании в днях |
| `title_len_chars` | длина текста | Длина заголовка в символах |
| `subtitle_len_chars` | длина текста | Длина подзаголовка в символах |
| `description_len_chars` | длина текста | Длина основного описания в символах |
| `meta_description_len_chars` | длина текста | Длина meta description в символах |
| `title_word_count` | длина текста | Число слов в заголовке |
| `subtitle_word_count` | длина текста | Число слов в подзаголовке |
| `description_word_count` | длина текста | Число слов в основном описании |
| `meta_description_word_count` | длина текста | Число слов в meta description |
| `description_has_link_word` | текстовый индикатор | Есть ли в описании слова/маркеры, указывающие на внешние площадки или ссылки |
| `category_grouped` | категория | Укрупнённая категория, используемая для дамми-переменных в моделях |
| `clean_text` | технический текст | Очищенная версия `description.text`, используемая в NLP-пайплайнах |

### 1.3 Текстовые признаки, извлечённые из описания

| Столбец | Роль | Описание |
| --- | --- | --- |
| `social_score` | словарный текстовый признак | Доля слов из словаря социальной/коллективной лексики |
| `gratitude_score` | словарный текстовый признак | Интенсивность благодарности: число паттернов благодарности на слово |
| `has_gratitude` | словарный текстовый признак | Бинарный флаг наличия хотя бы одного выражения благодарности |
| `we_count` | местоимения | Абсолютное число местоимений группы `мы/наш` |
| `i_count` | местоимения | Абсолютное число местоимений группы `я/мой` |
| `we_ratio` | местоимения | Доля `мы`-местоимений от общего числа слов |
| `i_ratio` | местоимения | Доля `я`-местоимений от общего числа слов |
| `we_vs_i` | местоимения | Отношение `we_count / (i_count + 1)` как индикатор коллективного стиля |
| `certainty_score` | модальность | Доля слов уверенности / определённости |
| `uncertainty_score` | модальность | Доля слов неопределённости / осторожности |
| `money_mentions` | конкретность | Число упоминаний денежных сумм в тексте |
| `number_density` | конкретность | Плотность чисел: число числовых выражений на слово |
| `has_specific_sum` | конкретность | Бинарный флаг наличия конкретной денежной суммы |
| `readability_flesch` | читаемость | Индекс Flesch Reading Ease для русского текста |
| `readability_fog` | читаемость | Индекс Gunning Fog |
| `readability_lix` | читаемость | Индекс LIX |
| `readability_avg` | читаемость | Сводный индекс читаемости: `(flesch - fog - lix) / 3` до последующей нормализации в моделях |
| `rubert_positive` | тональность | Вероятность позитивной тональности по RuBERT |
| `rubert_negative` | тональность | Вероятность негативной тональности по RuBERT |
| `topic_0` | тема | Вероятность принадлежности текста к латентной теме LDA №0 |
| `topic_1` | тема | Вероятность принадлежности текста к латентной теме LDA №1 |
| `topic_2` | тема | Вероятность принадлежности текста к латентной теме LDA №2 |
| `topic_3` | тема | Вероятность принадлежности текста к латентной теме LDA №3 |

---

## 2. Какие признаки участвуют в каких моделях

### 2.1 Производные признаки, создаваемые внутри модельных скриптов

Эти признаки не записаны отдельными колонками в `projects_sample_with_text.xlsx`, но создаются в коде перед обучением моделей:

| Производный признак | Как строится | Где используется |
| --- | --- | --- |
| `log_goal` | `log1p(card.targetAmount.value)` | `econometrics.py`, `ml_models.py`, `interpretability.py` |
| `log_text_length` | `log1p(description_len_chars)` | `econometrics.py`, `ml_models.py`, `interpretability.py` |
| `has_video` | `1`, если `video_count > 0`, иначе `0` | `econometrics.py`, `ml_models.py`, `interpretability.py` |
| `cat_*` | дамми-переменные из `category_grouped` | `econometrics.py`, `ml_models.py` |
| `funding_pct` | копия `funding_ratio` с отсечением снизу по нулю | `econometrics.py` |
| `log_funding_pct` | `log1p(funding_pct)` | `econometrics.py` при опции `--log-funding-pct` |

### 2.2 Эконометрические модели (`econometrics.py`)

Целевые переменные:

- Logit: `is_successful`
- OLS / Tobit: `funding_pct` или `log_funding_pct`

Базовые контролирующие признаки:

- `log_goal`
- `campaign_duration_days`
- `counts.newsCount`
- `counts.commentsCount`
- `card.author.campaignsAmount`
- `has_video`
- `log_text_length`
- `cat_*` из `category_grouped`

Спецификации моделей:

| Модель | Какие признаки входят |
| --- | --- |
| `M0 (controls)` | Только базовые контролирующие признаки |
| `M1 (+readability)` | `M0` + `readability_avg` + `description_word_count` |
| `M2 (+sentiment)` | `M0` + `rubert_positive` + `rubert_negative` |
| `M3 (+LIWC)` | `M0` + `social_score` + `gratitude_score` + `we_ratio` + `i_ratio` + `certainty_score` + `uncertainty_score` |
| `M_full` | `M0` + все признаки из `M1`, `M2` и `M3` |
| `M_lasso` | Тот же кандидатный набор, что и `M_full`, но с L1-регуляризацией |

Признаки из `projects_sample_with_text.xlsx`, которые **не участвуют** в эконометрических моделях напрямую:

- `image_count`, `external_link_count`, `we_count`, `i_count`, `we_vs_i`
- `money_mentions`, `number_density`, `has_specific_sum`
- `topic_0`, `topic_1`, `topic_2`, `topic_3`
- все URL, ID, сырые тексты и текстовые длины, кроме `description_word_count` и производного `log_text_length`

### 2.3 ML-модели (`ml_models.py`)

Целевая переменная:

- `is_successful`

Сравниваются модели:

- `LogReg_L1`
- `RandomForest`
- `GradBoost`
- опционально `LightGBM`, если библиотека установлена
- опционально `XGBoost`, если библиотека установлена

Наборы признаков:

| Набор | Какие признаки входят |
| --- | --- |
| `A (controls + cat)` | `log_goal`, `campaign_duration_days`, `counts.newsCount`, `counts.commentsCount`, `card.author.campaignsAmount`, `has_video`, `log_text_length`, `image_count`, `external_link_count` + `cat_*` |
| `B (text only)` | `social_score`, `gratitude_score`, `we_ratio`, `i_ratio`, `certainty_score`, `uncertainty_score`, `money_mentions`, `number_density`, `has_specific_sum`, `readability_avg`, `rubert_positive`, `rubert_negative` |
| `C (controls + cat + text)` | Объединение наборов `A` и `B` |

Фактические оговорки по текущей реализации:

- `topic_0`–`topic_3` в `ml_models.py` **не участвуют**.
- `clean_text` в текущей реализации `ml_models.py` тоже **не участвует**, несмотря на комментарий про `LDATransformer`.
- В коде перечислен `rubert_neutral`, но в файле `projects_sample_with_text.xlsx` этого столбца нет, поэтому фактически он в матрицу признаков не попадает.

### 2.4 Интерпретационные модели (`interpretability.py`)

Целевая переменная:

- `is_successful`

Обучаются и интерпретируются:

- `RandomForest`
- опционально `LightGBM`

Фактический набор признаков после фильтрации по наличию колонок:

- `log_goal`
- `campaign_duration_days`
- `counts.newsCount`
- `counts.commentsCount`
- `card.author.campaignsAmount`
- `has_video`
- `log_text_length`
- `image_count`
- `external_link_count`
- `social_score`
- `gratitude_score`
- `we_ratio`
- `certainty_score`
- `uncertainty_score`
- `readability_avg`
- `rubert_positive`
- `rubert_negative`
- `topic_0`
- `topic_1`
- `topic_2`
- `topic_3`

Оговорки:

- `i_ratio`, `money_mentions`, `number_density`, `has_specific_sum` здесь не используются.
- В коде перечислены `rubert_neutral` и `topic_4`, но из-за отсутствия этих колонок в `.xlsx` они автоматически отфильтровываются и фактически не участвуют.

### 2.5 TF-IDF-модели на уровне слов (`shap_tfidf.py`)

Этот скрипт не использует табличные признаки вроде `social_score` или `readability_avg`. Он строит отдельное текстовое представление:

- входной столбец: `clean_text`
- целевая переменная: `is_successful`
- признаки: TF-IDF по словам и биграммам
- модели: `LogisticRegression(L1)` и `LightGBM`

То есть `shap_tfidf.py` работает не с заранее рассчитанными столбцами-признаками, а с векторизацией текста заново.

### 2.6 Что не подаётся напрямую в текущие табличные модели

Колонки можно разделить на несколько типов неучастия:

- только целевые переменные: `funding_ratio`, `is_successful`
- только источник для производных модельных признаков: `card.targetAmount.value`, `video_count`, `description_len_chars`, `category_grouped`
- только источник для отдельной TF-IDF-модели: `clean_text`
- используются в описательной статистике / EDA, но не подаются напрямую в текущие предиктивные табличные модели: `has_gratitude`, `we_count`, `i_count`, `we_vs_i`, `readability_flesch`, `readability_fog`, `readability_lix`

Сырые поля, которые в текущих модельных скриптах не участвуют напрямую:

- `project_key`, `sourceUrl`
- `card.title`, `card.subtitle`, `description.text`, `meta.description`
- `card.collectedAmount.value`, `card.daysToFinish`, `card.startAt`, `card.finishAt`
- `card.region`, `card.mainCategory.tagName`, `card.author.id`
- `card.links.vk_url`, `card.links.telegram_url`, `card.links.author_site_url`
- `counts.participantsCount`, `counts.purchasesCount`
- `rewards.totalRewards`, `reward_count`
- `title_len_chars`, `subtitle_len_chars`, `meta_description_len_chars`
- `title_word_count`, `subtitle_word_count`, `meta_description_word_count`
- `description_has_link_word`

---

## 3. Быстрая навигация по группам признаков

Если нужен короткий ориентир:

- исходные поля платформы: `project_key` ... `rewards.totalRewards`
- инженерные кампанийные признаки: `image_count` ... `clean_text`
- словарные признаки: `social_score`, `gratitude_score`, `has_gratitude`, `we_*`, `i_*`, `certainty_score`, `uncertainty_score`
- признаки конкретности: `money_mentions`, `number_density`, `has_specific_sum`
- читаемость: `readability_flesch`, `readability_fog`, `readability_lix`, `readability_avg`
- тональность: `rubert_positive`, `rubert_negative`
- темы: `topic_0` ... `topic_3`
