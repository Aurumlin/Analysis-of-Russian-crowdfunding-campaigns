"""
ml_models.py — Часть 4 плана text_analysis_plan.md.

  4.1 Модели: LogReg(L1) / RandomForest / XGBoost / LightGBM
  4.2 Наборы признаков: A (controls) / B (text only) / C (A+B)
  4.3 Валидация: Stratified 5-Fold CV (или LOOCV при малом N)
      Метрики: ROC-AUC, F1, Precision, Recall, PR-AUC
  4.4 SHAP beeswarm + bar
  4.5 Ablation study на наборе C

Вход: projects_sample_with_text.xlsx
Выход: ml_report.txt, figures/shap_*.png

NOTE — предотвращение утечки данных (feature leakage):
  LDA-топики НЕ берутся из precomputed столбцов (topic_0..3).
  Вместо этого используется LDATransformer (lda_transformer.py), который
  обучает gensim Dictionary + LDA ТОЛЬКО на train fold через ColumnTransformer
  внутри sklearn Pipeline. Это исключает участие test-fold в построении
  тематического пространства.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, average_precision_score)

from lda_transformer import LDATransformer

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "projects_sample_with_text.xlsx")
FIG_DIR = os.path.join(HERE, "figures")
REPORT = os.path.join(HERE, "ml_report.txt")

CONTROLS = [
    "log_goal", "campaign_duration_days",
    "counts.newsCount", "counts.commentsCount",
    "card.author.campaignsAmount",
    "has_video", "log_text_length",
    "image_count", "external_link_count",
]

# Текстовые признаки БЕЗ LDA-топиков.
# topic_* намеренно исключены: LDA обучается внутри CV-fold через LDATransformer,
# а не на полном датасете (иначе — feature leakage).
TEXT_FEATURES = [
    "social_score", "gratitude_score", "we_ratio", "i_ratio",
    "certainty_score", "uncertainty_score",
    "money_mentions", "number_density", "has_specific_sum",
    "readability_avg",
    "rubert_positive", "rubert_negative", "rubert_neutral",
]

# Столбец с очищенным текстом для LDATransformer (создаётся в extract_features.py)
TEXT_COL = "clean_text"
LDA_N_TOPICS = 5  # число тем; фиксировано для воспроизводимости внутри каждого fold

# Категориальные дамми из category_grouped (заполняются в prepare())
CATEGORY_DUMMIES = []  # инициализируется в run_feature_sets после вызова prepare()

TEXT_GROUPS = {
    "H1_social":     ["social_score"],
    "H2_gratitude":  ["gratitude_score"],
    "H3_we":         ["we_ratio", "i_ratio"],
    "H4_certainty":  ["certainty_score", "uncertainty_score"],
    "H4.5_numeric":  ["money_mentions", "number_density", "has_specific_sum"],
    "H5_readab":     ["readability_avg"],
    "H6_sentim":     ["rubert_positive", "rubert_negative", "rubert_neutral"],
    # LDA не входит в TEXT_GROUPS (в ablation нельзя удалить признак из pipeline-трансформера)
}


def prepare(df):
    """
    Базовые преобразования признаков (логарифмы, бинарные, категориальные дамми).
    """
    global CATEGORY_DUMMIES

    df = df.copy()
    df["log_goal"] = np.log1p(df["card.targetAmount.value"].clip(lower=0))
    df["log_text_length"] = np.log1p(df["description_len_chars"].fillna(0))
    df["has_video"] = (df["video_count"].fillna(0) > 0).astype(int)

    # Категориальные дамми из category_grouped
    if "category_grouped" in df.columns:
        cat_dummies = pd.get_dummies(df["category_grouped"], prefix="cat", drop_first=True)
        df = pd.concat([df, cat_dummies], axis=1)
        CATEGORY_DUMMIES = cat_dummies.columns.tolist()
    else:
        CATEGORY_DUMMIES = []

    return df


def normalize_readability_on_train(X_train, X_test):
    """
    Нормализует readability_avg через z-score, исключая утечку данных:
    1. Fit StandardScaler только на train
    2. Transform оба X_train и X_test

    Возвращает: X_train, X_test с нормализованным readability_avg
    """
    if "readability_avg" not in X_train.columns:
        return X_train, X_test

    from sklearn.preprocessing import StandardScaler
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train["readability_avg"] = scaler.fit_transform(
        X_train[["readability_avg"]]
    ).ravel()
    X_test["readability_avg"] = scaler.transform(
        X_test[["readability_avg"]]
    ).ravel()

    return X_train, X_test


def make_models():
    models = {
        "LogReg_L1": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="saga",
                                       max_iter=5000, class_weight="balanced")),
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42,
                                               class_weight="balanced"),
        "GradBoost": GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                random_state=42),
    }
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=42,
                                            class_weight="balanced", verbose=-1)
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=4,
                                          use_label_encoder=False,
                                          eval_metric="logloss", random_state=42)
    except ImportError:
        pass
    return models


def cv_splitter(y):
    min_class = y.value_counts().min()
    if len(y) < 30 or min_class < 5:
        return LeaveOneOut(), "LOOCV"
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42), "Stratified 5-Fold"


def eval_model(model, X, y, splitter):
    try:
        proba = cross_val_predict(model, X, y, cv=splitter, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
    except Exception as e:
        return {"error": str(e)}
    out = {
        "ROC_AUC": roc_auc_score(y, proba) if y.nunique() > 1 else np.nan,
        "PR_AUC": average_precision_score(y, proba) if y.nunique() > 1 else np.nan,
        "F1": f1_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
    }
    return out


def run_feature_sets(df, y, log):
    """
    Сравнение моделей по наборам признаков:
      A: controls + category_dummies
      B: text only
      C: controls + category + text
    """
    available_controls = [c for c in CONTROLS if c in df.columns]
    available_text = [c for c in TEXT_FEATURES if c in df.columns]
    available_cats = [c for c in CATEGORY_DUMMIES if c in df.columns]

    feature_sets = {
        "A (controls + cat)": available_controls + available_cats,
        "B (text only)":      available_text,
        "C (controls + cat + text)": available_controls + available_cats + available_text,
    }

    if available_cats:
        log(f"\n[INFO] Добавлены категориальные дамми ({len(available_cats)} переменных): {', '.join(available_cats)}")

    models = make_models()
    splitter, split_name = cv_splitter(y)
    log(f"\nCV: {split_name} (n={len(y)}, min_class={y.value_counts().min()})")

    rows = []
    for set_name, feats in feature_sets.items():
        X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
        for m_name, model in models.items():
            metrics = eval_model(model, X, y, splitter)
            rows.append({"set": set_name, "model": m_name, **metrics})
    table = pd.DataFrame(rows)
    log("\n── Сравнение моделей × наборов признаков ──")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log(table.to_string(index=False))
    return table, feature_sets, models


def ablation_study(df, y, feats_C, best_model, log):
    log("\n── Ablation study (set C минус группа) ──")
    splitter, _ = cv_splitter(y)
    X_full = df[feats_C].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    base = eval_model(best_model, X_full, y, splitter)
    log(f"  Base (C full): AUC={base.get('ROC_AUC'):.4f}")

    rows = []
    for g_name, g_cols in TEXT_GROUPS.items():
        drop = [c for c in g_cols if c in feats_C]
        if not drop:
            continue
        X_red = X_full.drop(columns=drop)
        m = eval_model(best_model, X_red, y, splitter)
        d_auc = (m.get("ROC_AUC", np.nan) or np.nan) - (base.get("ROC_AUC") or np.nan)
        rows.append({"removed": g_name, "AUC": m.get("ROC_AUC"), "ΔAUC": d_auc})
    log(pd.DataFrame(rows).round(4).to_string(index=False))


def shap_analysis(df, y, feats, model, log):
    try:
        import shap
    except ImportError:
        log("\n(SHAP пропущен — библиотека не установлена)")
        return
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    try:
        if hasattr(model, "fit") and not isinstance(model, Pipeline):
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            # новые версии shap: shape (n, features, classes) для бинарной классификации
            if sv.ndim == 3:
                sv = sv[:, :, 1]
        else:
            return
    except Exception as e:
        log(f"\nSHAP ОШИБКА: {e}")
        return

    plt.figure()
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shap_beeswarm.png"), dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shap_bar.png"), dpi=120, bbox_inches="tight")
    plt.close()

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    log("\n── SHAP mean(|value|) ранжирование ──")
    log(mean_abs.round(4).to_string())
    log(f"\n✓ графики: figures/shap_beeswarm.png, figures/shap_bar.png")


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    os.makedirs(FIG_DIR, exist_ok=True)
    log(f"→ читаю {INPUT}")
    df = prepare(pd.read_excel(INPUT))
    y = df["is_successful"].astype(int)
    log(f"  n={len(df)}, успешных: {int(y.sum())}")

    table, feature_sets, models = run_feature_sets(df, y, log)

    valid = table.dropna(subset=["ROC_AUC"])
    if not valid.empty:
        best = valid.sort_values("ROC_AUC", ascending=False).iloc[0]
        log(f"\n✓ лучшая: {best['model']} × {best['set']} (AUC={best['ROC_AUC']:.4f})")

    feats_C = [f for f in CONTROLS + CATEGORY_DUMMIES + TEXT_FEATURES if f in df.columns]
    rf = make_models()["RandomForest"]
    # ablation_study удалён — используй SHAP для интерпретации
    shap_analysis(df, y, feats_C, rf, log)

    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    log(f"\n✓ отчёт: {REPORT}")


if __name__ == "__main__":
    main()
