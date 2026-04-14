"""
extract_features.py — Шаг 1 плана text_analysis_plan.md.

Извлекает текстовые признаки из projects_sample.xlsx:
  1.1 social_score
  1.2 gratitude_score, has_gratitude
  1.3 we_count, i_count, we_ratio, i_ratio, we_vs_i
  1.4 certainty_score, uncertainty_score
  1.5 readability_flesch/fog/lix/avg
  1.6 rubert_positive/negative/neutral (RuBERT Sentiment)
  1.7 LDA topic_* (gensim с автоматическим подбором K по Coherence)

ПРИМЕЧАНИЕ: категориальные дамми из category_grouped создаются в ml_models.py,
не здесь. Это позволяет избежать утечки данных при кросс-валидации.

Запуск:  python3 extract_features.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from text_features import extract_all_text_features, clean_text  # 1.1–1.5
import dicts

INPUT_FILE = os.path.join(HERE, "projects_sample.xlsx")
OUTPUT_FILE = os.path.join(HERE, "projects_sample_with_text.xlsx")
TEXT_COL = "description.text"
CLEAN_COL = "clean_text"  # создаётся внутри extract_all_text_features


# ─────────────────────────────────────────────────────────────────────────────
# 1.6 RuBERT Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def add_rubert_sentiment(df: pd.DataFrame, text_col: str = CLEAN_COL) -> pd.DataFrame:
    """Добавляет rubert_positive/negative/neutral. Агрегация = mean по предложениям."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError as e:
        print(f"[RuBERT] пропущено — нет зависимости: {e}")
        for c in ("rubert_positive", "rubert_negative", "rubert_neutral"):
            df[c] = np.nan
        return df

    model_name = "blanchefort/rubert-base-cased-sentiment"
    print(f"[RuBERT] загружаю {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    id2label = model.config.id2label  # {0: 'NEUTRAL', 1: 'POSITIVE', 2: 'NEGATIVE'}
    label_map = {v.upper(): k for k, v in id2label.items()}

    def score_text(text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {"rubert_positive": np.nan,
                    "rubert_negative": np.nan,
                    "rubert_neutral": np.nan}
        try:
            sents = nltk.sent_tokenize(text, language="russian")
        except Exception:
            sents = [text]
        sents = [s[:512] for s in sents if s.strip()][:200]
        if not sents:
            return {"rubert_positive": np.nan,
                    "rubert_negative": np.nan,
                    "rubert_neutral": np.nan}

        probs_acc = []
        with torch.no_grad():
            for i in range(0, len(sents), 16):
                batch = sents[i:i + 16]
                enc = tok(batch, padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                probs_acc.append(probs)
        probs = np.vstack(probs_acc).mean(axis=0)
        return {
            "rubert_positive": float(probs[label_map["POSITIVE"]]),
            "rubert_negative": float(probs[label_map["NEGATIVE"]]),
            "rubert_neutral":  float(probs[label_map["NEUTRAL"]]),
        }

    print(f"[RuBERT] обработка {len(df)} текстов...")
    scores = [score_text(t) for t in df[text_col]]
    return pd.concat([df, pd.DataFrame(scores, index=df.index)], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 1.7 LDA
# ─────────────────────────────────────────────────────────────────────────────

def add_lda_topics(df: pd.DataFrame,
                   text_col: str = CLEAN_COL,
                   num_topics: int = 5) -> pd.DataFrame:
    """Добавляет topic_0..topic_{K-1} — распределение по темам."""
    try:
        from gensim.corpora import Dictionary
        from gensim.models import LdaModel
    except ImportError as e:
        print(f"[LDA] пропущено — нет gensim: {e}")
        return df

    from text_features import tokenize, lemmatize_tokens, clean_text_for_lda

    RU_STOP = {
        "и", "в", "на", "не", "что", "с", "по", "а", "как", "это", "у", "за",
        "но", "из", "от", "к", "о", "же", "то", "для", "бы", "так", "вот",
        "был", "быть", "есть", "или", "еще", "уже", "мы", "вы", "он", "она",
        "они", "я", "ты", "наш", "ваш", "свой", "этот", "тот", "весь", "все",
    }

    docs = []
    for t in df[text_col]:
        # Применяем LDA-специфичную чистку (удаляет цифры, пунктуацию)
        t_clean = clean_text_for_lda(t)
        tokens = tokenize(t_clean)
        if not tokens:
            docs.append([])
            continue
        lemmas = lemmatize_tokens([w.lower() for w in tokens])
        lemmas = [w for w in lemmas if len(w) > 2 and w not in RU_STOP]
        docs.append(lemmas)

    if all(len(d) == 0 for d in docs):
        print("[LDA] пустой корпус, пропускаю")
        return df

    dictionary = Dictionary(docs)
    # для маленьких выборок убираем жёсткие фильтры
    if len(docs) > 30:
        dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(d) for d in docs]

    print(f"[LDA] обучение LDA (K={num_topics}) на {len(docs)} документах...")
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                   passes=10, random_state=42)

    topic_matrix = np.zeros((len(docs), num_topics))
    for i, bow in enumerate(corpus):
        for tid, p in lda.get_document_topics(bow, minimum_probability=0.0):
            topic_matrix[i, tid] = p

    topic_df = pd.DataFrame(
        topic_matrix,
        columns=[f"topic_{i}" for i in range(num_topics)],
        index=df.index,
    )
    # топ-слова для интерпретации
    print("[LDA] топ-слова по темам:")
    for i in range(num_topics):
        top = [w for w, _ in lda.show_topic(i, topn=8)]
        print(f"  topic_{i}: {', '.join(top)}")

    return pd.concat([df, topic_df], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def print_dicts():
    sep = "─" * 60
    print(f"\n{sep}")
    print("СЛОВАРИ (из dicts.py — редактируй там)")
    print(sep)
    print(f"\n[1.1] SOCIAL_WORDS ({len(dicts.SOCIAL_WORDS)} слов):")
    print("  " + ", ".join(sorted(dicts.SOCIAL_WORDS)))

    print(f"\n[1.2] GRATITUDE_ROOTS ({len(dicts.GRATITUDE_ROOTS)} паттернов, regex):")
    for p in dicts.GRATITUDE_ROOTS:
        print(f"  {p}")

    print(f"\n[1.3] WE_WORDS ({len(dicts.WE_WORDS)} форм):")
    print("  " + ", ".join(sorted(dicts.WE_WORDS)))
    print(f"\n[1.3] I_WORDS ({len(dicts.I_WORDS)} форм):")
    print("  " + ", ".join(sorted(dicts.I_WORDS)))

    print(f"\n[1.4] CERTAINTY_WORDS ({len(dicts.CERTAINTY_WORDS)} слов):")
    print("  " + ", ".join(sorted(dicts.CERTAINTY_WORDS)))
    print(f"\n[1.4] UNCERTAINTY_WORDS ({len(dicts.UNCERTAINTY_WORDS)} слов):")
    print("  " + ", ".join(sorted(dicts.UNCERTAINTY_WORDS)))
    print(f"\n{sep}\n")


def validate_dictionary_coverage(df: pd.DataFrame) -> None:
    """
    Проверяет, какой % текстов содержат слова из словарей.

    ВАЖНО: если покрытие < 10% (0.1), словарь почти не работает на этом корпусе
    и нужно либо расширить словарь, либо пересмотреть подход.
    """
    print(f"\n{'─' * 60}")
    print("ВАЛИДАЦИЯ: Покрытие словарей")
    print("─" * 60)

    coverage = {
        "social_score":      (df["social_score"] > 0).sum() / len(df),
        "gratitude_score":   (df["gratitude_score"] > 0).sum() / len(df),
        "certainty_score":   (df["certainty_score"] > 0).sum() / len(df),
        "uncertainty_score": (df["uncertainty_score"] > 0).sum() / len(df),
    }

    for key, pct in coverage.items():
        emoji = "⚠️ " if pct < 0.1 else "✓"
        print(f"{emoji} {key:20s}: {pct*100:5.1f}% ({(df[key] > 0).sum():3d}/{len(df)} текстов)")

    print("\n⚠️  ИНТЕРПРЕТАЦИЯ:")
    print("  Если покрытие < 10% — словарь почти не работает на данном корпусе.")
    print("  Действия: расширить словарь, проверить логику поиска или пересмотреть метод.")
    print(f"{'─' * 60}\n")


def main():
    print_dicts()
    print(f"→ читаю {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)
    print(f"  shape: {df.shape}")

    print("→ 1.1–1.5: social / gratitude / collectivism / certainty / readability")
    df = extract_all_text_features(df, text_col=TEXT_COL)

    print("→ 1.6: RuBERT sentiment")
    df = add_rubert_sentiment(df, text_col=CLEAN_COL)

    print("→ 1.7: LDA topic distribution")
    df = add_lda_topics(df, text_col=CLEAN_COL, num_topics=5)

    # ─────────────────────────────────────────────────────────────────────────────
    # ВАЖНО: Удаляем рубертовский neutral и topic_4 для избежания идеальной коллинеарности
    # ─────────────────────────────────────────────────────────────────────────────
    # rubert_positive + rubert_negative + rubert_neutral = 1.0 (всегда)
    # topic_0 + topic_1 + topic_2 + topic_3 + topic_4 = 1.0 (всегда)
    # Дропаем один столбец из каждой группы перед регрессией
    if "rubert_neutral" in df.columns:
        df.drop(columns=["rubert_neutral"], inplace=True)
    if "topic_4" in df.columns:
        df.drop(columns=["topic_4"], inplace=True)

    new_cols = [
        "social_score",
        "gratitude_score", "has_gratitude",
        "we_count", "i_count", "we_ratio", "i_ratio", "we_vs_i",
        "certainty_score", "uncertainty_score",
        "money_mentions", "number_density", "has_specific_sum",
        "readability_flesch", "readability_fog", "readability_lix", "readability_avg",
        "rubert_positive", "rubert_negative",  # neutral удалён
    ] + [f"topic_{i}" for i in range(4)]  # topic_4 удалён
    existing = [c for c in new_cols if c in df.columns]
    print("\n=== СТАТИСТИКА НОВЫХ ПРИЗНАКОВ ===")
    print(df[existing].describe().T[["mean", "std", "min", "max"]].round(4))

    # ВАЛИДАЦИЯ: Покрытие словарей
    validate_dictionary_coverage(df)

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n✓ сохранено: {OUTPUT_FILE}")
    print("\n[ВНИМАНИЕ] Удалены rubert_neutral и topic_4 для избежания идеальной коллинеарности.")


if __name__ == "__main__":
    main()
