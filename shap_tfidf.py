"""
shap_tfidf.py — SHAP-анализ на уровне слов через TF-IDF.

Pipeline:
  1. Очистка + лемматизация description.text
  2. TF-IDF (top-500 uni+bigrams, min_df=2)
  3. Модель A: LogisticRegression(L1)  → LinearExplainer
  4. Модель B: LightGBM                → TreeExplainer
  5. Графики:
       figures/tfidf_logreg_coef.png   — коэффициенты LogReg
       figures/tfidf_shap_logreg.png   — SHAP beeswarm (LogReg)
       figures/tfidf_shap_lgbm.png     — SHAP beeswarm (LightGBM)
       figures/tfidf_shap_lgbm_bar.png — SHAP bar (LightGBM)
  6. shap_words_report.csv            — топ-слов с SHAP-значениями

Вход: projects_sample_with_text.xlsx
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "projects_sample_with_text.xlsx")
FIG_DIR = os.path.join(HERE, "figures")
REPORT_CSV = os.path.join(HERE, "shap_words_report.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Параметры — меняй здесь
# ─────────────────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 500    # сколько слов брать в словарь TF-IDF
TFIDF_MIN_DF = 2            # минимум документов, в которых встречается слово
                            # (на sample=10 ставь 1, на полном датасете — 5+)
TFIDF_NGRAM = (1, 2)        # (1,1)=только слова, (1,2)=слова+биграммы
TOP_WORDS = 20              # сколько топ-слов показывать на графиках
TARGET = "is_successful"
TEXT_COL = "clean_text"     # очищенный текст (создан в extract_features.py)


# ─────────────────────────────────────────────────────────────────────────────
# Лемматизация для TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pymorphy3 as _pm
except ImportError:
    import pymorphy2 as _pm

morph = _pm.MorphAnalyzer()

RU_STOP = {
    "и", "в", "на", "не", "что", "с", "по", "а", "как", "это", "у", "за",
    "но", "из", "от", "к", "о", "же", "то", "для", "бы", "так", "вот",
    "был", "быть", "есть", "или", "еще", "уже", "мы", "вы", "он", "она",
    "они", "я", "ты", "наш", "ваш", "свой", "этот", "тот", "весь", "все",
    "при", "до", "со", "об", "без", "через", "над", "под", "между", "после",
    "также", "который", "которая", "которые", "которого",
}


def lemmatize_for_tfidf(text: str) -> str:
    """Лемматизирует текст, убирает стоп-слова, возвращает строку лемм."""
    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = re.findall(r"[а-яёА-ЯЁ]+", text.lower())
    lemmas = []
    for t in tokens:
        if len(t) < 3:
            continue
        lemma = morph.parse(t)[0].normal_form
        if lemma not in RU_STOP:
            lemmas.append(lemma)
    return " ".join(lemmas)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF + модели
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_matrix(texts: pd.Series):
    from sklearn.feature_extraction.text import TfidfVectorizer
    min_df = min(TFIDF_MIN_DF, max(1, len(texts) // 5))
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=min_df,
        ngram_range=TFIDF_NGRAM,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    print(f"  TF-IDF: {X.shape[1]} признаков, {X.shape[0]} документов (min_df={min_df})")
    return X, vec


# ─────────────────────────────────────────────────────────────────────────────
# Модель A: LogReg + LinearExplainer
# ─────────────────────────────────────────────────────────────────────────────

def run_logreg_shap(X_sparse, y: np.ndarray, feature_names: list):
    import shap
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MaxAbsScaler

    print("\n→ Модель A: LogisticRegression(L1)")
    scaler = MaxAbsScaler()
    Xs = scaler.fit_transform(X_sparse)

    model = LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000,
        class_weight="balanced", C=1.0,
    )
    model.fit(Xs, y)

    # Коэффициенты LogReg (без SHAP — прямая интерпретация)
    coef = pd.Series(model.coef_.ravel(), index=feature_names)
    top_pos = coef.nlargest(TOP_WORDS)
    top_neg = coef.nsmallest(TOP_WORDS)
    top_all = pd.concat([top_pos, top_neg]).sort_values()

    fig, ax = plt.subplots(figsize=(8, max(6, len(top_all) * 0.3)))
    colors = ["#d73027" if v > 0 else "#4575b4" for v in top_all]
    top_all.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"LogReg(L1) коэффициенты: топ-{TOP_WORDS} слов\n"
                 f"красный = ↑ успех, синий = ↓ успех")
    ax.set_xlabel("Коэффициент")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "tfidf_logreg_coef.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")

    # SHAP LinearExplainer
    print("  вычисляю SHAP (LinearExplainer)...")
    X_dense = Xs.toarray() if hasattr(Xs, "toarray") else Xs
    background = shap.maskers.Independent(X_dense, max_samples=min(50, len(y)))
    explainer = shap.LinearExplainer(model, background)
    sv = explainer.shap_values(X_dense)
    # бинарная классификация: берём класс 1 (успех)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)

    shap_df = pd.DataFrame(sv, columns=feature_names)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(6, TOP_WORDS * 0.3)))
    top_feats = mean_abs.head(TOP_WORDS).index.tolist()
    shap.summary_plot(sv[:, [feature_names.index(f) for f in top_feats]],
                      X_dense[:, [feature_names.index(f) for f in top_feats]],
                      feature_names=top_feats, show=False, plot_size=None)
    plt.title(f"SHAP (LogReg): топ-{TOP_WORDS} слов")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "tfidf_shap_logreg.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")

    return coef, shap_df


# ─────────────────────────────────────────────────────────────────────────────
# Модель B: LightGBM + TreeExplainer
# ─────────────────────────────────────────────────────────────────────────────

def run_lgbm_shap(X_sparse, y: np.ndarray, feature_names: list):
    import shap
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("  LightGBM не установлен, пропускаю модель B")
        return None

    print("\n→ Модель B: LightGBM")
    X_dense = X_sparse.toarray() if hasattr(X_sparse, "toarray") else X_sparse

    model = LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    model.fit(X_dense, y)

    print("  вычисляю SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_dense)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    shap_df = pd.DataFrame(sv, columns=feature_names)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top_feats = mean_abs.head(TOP_WORDS).index.tolist()
    top_idx = [feature_names.index(f) for f in top_feats]

    # beeswarm
    fig, ax = plt.subplots(figsize=(8, max(6, TOP_WORDS * 0.35)))
    shap.summary_plot(sv[:, top_idx], X_dense[:, top_idx],
                      feature_names=top_feats, show=False, plot_size=None)
    plt.title(f"SHAP (LightGBM): топ-{TOP_WORDS} слов")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "tfidf_shap_lgbm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")

    # bar
    fig, ax = plt.subplots(figsize=(8, max(6, TOP_WORDS * 0.35)))
    shap.summary_plot(sv[:, top_idx], X_dense[:, top_idx],
                      feature_names=top_feats, plot_type="bar",
                      show=False, plot_size=None)
    plt.title(f"SHAP mean(|value|) (LightGBM): топ-{TOP_WORDS} слов")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "tfidf_shap_lgbm_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")

    return shap_df


# ─────────────────────────────────────────────────────────────────────────────
# Сводный CSV-отчёт
# ─────────────────────────────────────────────────────────────────────────────

def save_report(feature_names, coef_logreg, shap_logreg, shap_lgbm):
    report = pd.DataFrame({"word": feature_names})
    if coef_logreg is not None:
        report["logreg_coef"] = coef_logreg.reindex(feature_names).values
    if shap_logreg is not None:
        report["shap_logreg_mean_abs"] = shap_logreg.abs().mean().reindex(feature_names).values
        report["shap_logreg_mean"] = shap_logreg.mean().reindex(feature_names).values
    if shap_lgbm is not None:
        report["shap_lgbm_mean_abs"] = shap_lgbm.abs().mean().reindex(feature_names).values
        report["shap_lgbm_mean"] = shap_lgbm.mean().reindex(feature_names).values

    report = report.sort_values("shap_lgbm_mean_abs" if shap_lgbm is not None
                                else "logreg_coef", ascending=False, key=abs)
    report.to_csv(REPORT_CSV, index=False)
    print(f"\n✓ отчёт: {REPORT_CSV}")

    print(f"\n── Топ-{TOP_WORDS} слов по SHAP (LightGBM) ──")
    cols = [c for c in ["word", "shap_lgbm_mean_abs", "shap_lgbm_mean", "logreg_coef"]
            if c in report.columns]
    print(report[cols].head(TOP_WORDS).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"→ читаю {INPUT}")
    df = pd.read_excel(INPUT)
    print(f"  shape: {df.shape}, успешных: {int(df[TARGET].sum())} / {len(df)}")

    if TEXT_COL not in df.columns:
        raise ValueError(f"Колонка '{TEXT_COL}' не найдена. "
                         "Сначала запусти extract_features.py")

    print("→ лемматизирую тексты для TF-IDF...")
    lemmatized = df[TEXT_COL].apply(lemmatize_for_tfidf)
    empty = (lemmatized.str.strip() == "").sum()
    if empty:
        print(f"  ! пустых текстов после лемматизации: {empty}")

    print("→ строю TF-IDF матрицу...")
    X_tfidf, vectorizer = build_tfidf_matrix(lemmatized)
    feature_names = vectorizer.get_feature_names_out().tolist()
    y = df[TARGET].astype(int).values

    coef_logreg, shap_logreg = run_logreg_shap(X_tfidf, y, feature_names)
    shap_lgbm = run_lgbm_shap(X_tfidf, y, feature_names)
    save_report(feature_names, coef_logreg, shap_logreg, shap_lgbm)


if __name__ == "__main__":
    main()
