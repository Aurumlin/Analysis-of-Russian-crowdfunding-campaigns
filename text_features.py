"""
text_features.py
Извлечение текстовых признаков для краудфандинговых кампаний Planeta.ru
"""

import re
import html
import math
import numpy as np
import pandas as pd
import nltk
import pyphen

try:
    import pymorphy3 as _pymorphy
except ImportError:
    import pymorphy2 as _pymorphy

morph = _pymorphy.MorphAnalyzer()

# ─────────────────────────────────────────────────────────────────────────────
# СЛОВАРИ — импортируются из dicts.py, редактируй там
# ─────────────────────────────────────────────────────────────────────────────
from dicts import (
    SOCIAL_WORDS, GRATITUDE_ROOTS,
    WE_WORDS, I_WORDS,
    CERTAINTY_WORDS, UNCERTAINTY_WORDS,
)

GRATITUDE_PATTERN = re.compile(
    "|".join(GRATITUDE_ROOTS), flags=re.IGNORECASE | re.UNICODE
)

# ─────────────────────────────────────────────────────────────────────────────
# ОЧИСТКА ТЕКСТА (РАЗДЕЛЬНАЯ для разных задач)
# ─────────────────────────────────────────────────────────────────────────────

def clean_text_for_liwc(text: str) -> str:
    """
    Для словарных признаков (1.1–1.4: social, gratitude, certainty, uncertainty):
    Оставляем только слова → цифры не нужны, пунктуация не нужна.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_readability(text: str) -> str:
    """
    Для индексов читаемости (1.5: Flesch, Fog, LIX):
    Сохраняем пунктуацию (нужна для разбивки на предложения)
    и цифры (влияют на длину слов/предложений).
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_rubert(text: str) -> str:
    """
    Для RuBERT (1.6: sentiment):
    Минимальная чистка — модель сама умеет обрабатывать
    пунктуацию, цифры, заглавные буквы.
    Удалять: HTML-теги, множественные пробелы (только).
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_lda(text: str) -> str:
    """
    Для LDA (1.7: topic modeling):
    Оставляем только слова, убираем цифры и пунктуацию.
    Лемматизация и фильтр стоп-слов делаются отдельно.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^а-яёА-ЯЁa-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    """
    DEPRECATED: используй clean_text_for_* функции для конкретной задачи.
    留 для совместимости вызывает clean_text_for_liwc().
    """
    return clean_text_for_liwc(text)


# ─────────────────────────────────────────────────────────────────────────────
# ПРЕПРОЦЕССИНГ
# ─────────────────────────────────────────────────────────────────────────────

def count_syllables_ru(word: str) -> int:
    """Считает слоги как количество гласных букв."""
    vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
    return sum(1 for ch in word if ch in vowels)


def tokenize(text: str) -> list[str]:
    """Простая токенизация: только буквенные токены."""
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"[а-яёА-ЯЁa-zA-Z]+", text)


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Лемматизация через pymorphy2."""
    return [morph.parse(t)[0].normal_form for t in tokens]


def split_sentences(text: str) -> list[str]:
    """Разбивка на предложения через nltk."""
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        return nltk.sent_tokenize(text, language="russian")
    except Exception:
        return re.split(r"[.!?]+", text)


# ─────────────────────────────────────────────────────────────────────────────
# 1.1 СОЦИАЛЬНОСТЬ
# ─────────────────────────────────────────────────────────────────────────────

def compute_social_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    lemmas = lemmatize_tokens(tokens)
    social_count = sum(1 for l in lemmas if l in SOCIAL_WORDS)
    return social_count / len(lemmas)


# ─────────────────────────────────────────────────────────────────────────────
# 1.2 БЛАГОДАРНОСТЬ
# ─────────────────────────────────────────────────────────────────────────────

def compute_gratitude(text: str) -> dict:
    """
    Благодарность: регулярные выражения в сыром тексте.

    УНИФИКАЦИЯ: знаменатель = len(tokens) (токены, не леммы).
    Это согласуется с we_ratio и другими признаками, нормализованными к словам.
    """
    if not isinstance(text, str):
        return {"gratitude_score": 0.0, "has_gratitude": 0}
    matches = GRATITUDE_PATTERN.findall(text)
    tokens = tokenize(text)
    word_count = len(tokens) if tokens else 1
    return {
        "gratitude_score": len(matches) / word_count,
        "has_gratitude": int(len(matches) > 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1.3 КОЛЛЕКТИВНОСТЬ
# ─────────────────────────────────────────────────────────────────────────────

def compute_collectivism(text: str) -> dict:
    """
    Коллективность: "мы" vs "я" местоимения.

    УНИФИКАЦИЯ:
      - we_ratio = мы_count / len(tokens)     [% мы-местоимений от всех слов]
      - i_ratio = я_count / len(tokens)       [% я-местоимений от всех слов]
      - we_vs_i = мы_count / (я_count + 1)    [безразмерная; осталась как была]

    Старое значение we_ratio (= мы / (мы + я)) недоступно прямо, но можно вычислить как:
      we_ratio_old = we_ratio / (we_ratio + i_ratio + 1e-6)
    """
    tokens = tokenize(text)
    if not tokens:
        return {
            "we_count": 0, "i_count": 0,
            "we_ratio": 0.0, "i_ratio": 0.0,
            "we_vs_i": 0.0,
        }

    lowered = [t.lower() for t in tokens]
    we_count = sum(1 for t in lowered if t in WE_WORDS)
    i_count = sum(1 for t in lowered if t in I_WORDS)

    n = len(tokens)  # знаменатель = все слова в тексте
    we_ratio = we_count / n
    i_ratio = i_count / n
    we_vs_i = we_count / (i_count + 1)  # безразмерная шкала

    return {
        "we_count": we_count,
        "i_count": i_count,
        "we_ratio": we_ratio,
        "i_ratio": i_ratio,
        "we_vs_i": we_vs_i,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1.4 УВЕРЕННОСТЬ / НЕУВЕРЕННОСТЬ
# ─────────────────────────────────────────────────────────────────────────────

def compute_certainty(text: str) -> dict:
    tokens = tokenize(text)
    if not tokens:
        return {"certainty_score": 0.0, "uncertainty_score": 0.0}
    lemmas = lemmatize_tokens(tokens)
    n = len(lemmas)
    certainty_count = sum(1 for l in lemmas if l in CERTAINTY_WORDS)
    uncertainty_count = sum(1 for l in lemmas if l in UNCERTAINTY_WORDS)
    return {
        "certainty_score": certainty_count / n,
        "uncertainty_score": uncertainty_count / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1.4.5 ЧИСЛОВЫЕ ПРИЗНАКИ (КОНКРЕТНОСТЬ ОПИСАНИЯ)
# ─────────────────────────────────────────────────────────────────────────────

def compute_numeric_features(text: str) -> dict:
    """
    Цифры в тексте сигнализируют о конкретности описания.

    Пример:
      "Мы напечатаем 500 книг для 10 школ" → очень конкретно (number_density высокая)
      "Мы сделаем много добра" → абстрактно (number_density низкая)

    Признаки:
      - money_mentions: число упоминаний денежных сумм (рубли, тысячи, млн)
      - number_density: % слов, являющихся цифрами (от всех слов)
      - has_specific_sum: бинарный флаг (есть ли конкретная сумма в рублях)
    """
    if not isinstance(text, str):
        return {
            "money_mentions": 0,
            "number_density": 0.0,
            "has_specific_sum": 0,
        }

    # Число упоминаний денежных сумм: "500 руб", "2.5 тыс", "1 млн₽"
    money_pattern = re.compile(
        r"\d[\d\s,.]*\s*(?:руб|рублей|р\.|₽|тыс|тысяч|млн|миллион|копейк)",
        re.IGNORECASE
    )
    # Любые числовые значения (целые и с точкой)
    numbers_pattern = re.compile(r"\b\d+(?:[.,]\d+)?\b")

    tokens = tokenize(text)
    n_tokens = max(len(tokens), 1)

    money_count = len(money_pattern.findall(text))
    number_count = len(numbers_pattern.findall(text))

    return {
        "money_mentions": money_count,
        "number_density": number_count / n_tokens,
        "has_specific_sum": int(bool(money_pattern.search(text))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1.5 ЧИТАЕМОСТЬ
# ─────────────────────────────────────────────────────────────────────────────

def compute_readability(text: str) -> dict:
    """Flesch Reading Ease (ru адаптация), Gunning Fog, LIX."""
    if not isinstance(text, str) or not text.strip():
        return {
            "readability_flesch": np.nan,
            "readability_fog": np.nan,
            "readability_lix": np.nan,
        }

    sentences = split_sentences(text)
    sentences = [s for s in sentences if s.strip()]
    n_sentences = max(len(sentences), 1)

    tokens = re.findall(r"[а-яёА-ЯЁa-zA-Z]+", text)
    n_words = len(tokens)
    if n_words == 0:
        return {
            "readability_flesch": np.nan,
            "readability_fog": np.nan,
            "readability_lix": np.nan,
        }

    n_syllables = sum(count_syllables_ru(w) for w in tokens)
    n_complex = sum(1 for w in tokens if count_syllables_ru(w) >= 3)
    n_long = sum(1 for w in tokens if len(w) > 6)

    asl = n_words / n_sentences        # avg sentence length
    asw = n_syllables / n_words        # avg syllables per word

    flesch = 206.835 - 1.015 * asl - 84.6 * asw
    fog = 0.4 * (asl + 100 * n_complex / n_words)
    lix = asl + 100 * n_long / n_words

    return {
        "readability_flesch": round(flesch, 4),
        "readability_fog": round(fog, 4),
        "readability_lix": round(lix, 4),
    }


def add_readability_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет readability_avg как простое среднее трёх сырых индексов (БЕЗ нормализации).

    ⚠️ ВАЖНО: инвертируем FOG и LIX, чтобы согласовать направление (выше = проще читать)

    Нормализацию (z-score) делает конечный пайплайн:
    - НИКОГДА в extract_features.py
    - В ml_models.py / econometrics.py / interpretability.py ПОСЛЕ train/test split
    - С СОХРАНЕНИЕМ параметров (mean, std) для применения к новым данным

    Это избегает проблем:
    ✓ Data leakage (test не видит train statistics)
    ✓ Воспроизводимость (можно добавлять новые данные без пересчёта всех старых)
    ✓ Монотонность (z-score не меняются при добавлении новых данных)
    """
    cols = ["readability_flesch", "readability_fog", "readability_lix"]
    df = df.copy()

    # Инвертировать FOG и LIX: они идут "в обратном направлении"
    # (выше значение = сложнее текст, а нам нужно выше = проще)
    df["readability_avg"] = (
        df["readability_flesch"] - df["readability_fog"] - df["readability_lix"]
    ) / 3.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ: всё сразу
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_text_features(df: pd.DataFrame, text_col: str = "description.text") -> pd.DataFrame:
    """
    Добавляет к df все текстовые признаки частей 1.1–1.5 + 1.4.5 (числовые).

    Использует раздельную чистку текста:
      - clean_text_for_liwc:       для социальности, благодарности, уверенности (1.1–1.4)
      - clean_text_for_readability: для индексов читаемости (1.5)
      - clean_text_for_lda:        для LDA (1.7, но используется в extract_features.py отдельно)

    Сохраняет чистый текст (для RuBERT в extract_features.py) в колонке 'clean_text'.
    """
    df = df.copy()
    # Используем минимальную чистку для RuBERT
    df["clean_text"] = df[text_col].apply(clean_text_for_rubert)

    results = []
    for orig_text in df[text_col]:
        row = {}

        # 1.1–1.4: словарные признаки (используем LIWC-чистку)
        text_liwc = clean_text_for_liwc(orig_text)
        row["social_score"] = compute_social_score(text_liwc)
        row.update(compute_gratitude(text_liwc))
        row.update(compute_collectivism(text_liwc))
        row.update(compute_certainty(text_liwc))

        # 1.4.5: числовые признаки (конкретность)
        row.update(compute_numeric_features(orig_text))

        # 1.5: читаемость (используем чистку с пунктуацией и цифрами)
        text_read = clean_text_for_readability(orig_text)
        row.update(compute_readability(text_read))

        results.append(row)

    feat_df = pd.DataFrame(results, index=df.index)
    out = pd.concat([df, feat_df], axis=1)
    out = add_readability_avg(out)
    return out
