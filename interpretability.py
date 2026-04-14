"""
interpretability.py — Расширенная интерпретация ML-моделей.

  0.  Train/test split (стратифицированный, 80/20)
  1.  Обучение RandomForest + LightGBM на train
  2.  Метрики на test
  3.  Визуализация числовых признаков (гистограммы, корреляции, boxplot по группам)
  4.  Permutation Importance (test)
  5.  Partial Dependence Plots + ICE (топ признаки)
  6.  TreeSHAP: beeswarm, bar, dependence plots
  7.  Mean SHAP по категориям (category_grouped)
  8.  LIME: объяснение нескольких примеров из test
  9.  GAM (pyGAM LogisticGAM): partial effects каждого признака
  10. Learning Curves (train size vs ROC-AUC)
  11. Ошибки по сегментам (категория, квартиль цели, has_video)

Вход:  projects_sample_with_text.xlsx
Выход: figures/interp_*.png, interpretability_report.txt
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "projects_sample_with_text.xlsx")
FIG_DIR = os.path.join(HERE, "figures")
REPORT = os.path.join(HERE, "interpretability_report.txt")

TARGET = "is_successful"
CAT_COL = "category_grouped"   # колонка с категорией проекта

CONTROLS = [
    "log_goal", "campaign_duration_days",
    "counts.newsCount", "counts.commentsCount",
    "card.author.campaignsAmount",
    "has_video", "log_text_length",
    "image_count", "external_link_count",
]
TEXT_FEATURES = [
    "social_score", "gratitude_score", "we_ratio",
    "certainty_score", "uncertainty_score",
    "readability_avg",
    "rubert_positive", "rubert_negative", "rubert_neutral",
    "topic_0", "topic_1", "topic_2", "topic_3", "topic_4",
]

# ─── сколько объяснений LIME рисовать ───────────────────────────────────────
LIME_N_EXAMPLES = 3
# ─── топ признаков для PDP/ICE и SHAP dependence ────────────────────────────
TOP_N = 6


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────

def savefig(name: str):
    path = os.path.join(FIG_DIR, f"interp_{name}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def log_print(lines: list, msg=""):
    print(msg)
    lines.append(str(msg))


# ─────────────────────────────────────────────────────────────────────────────
# Подготовка данных
# ─────────────────────────────────────────────────────────────────────────────

def prepare(df: pd.DataFrame):
    df = df.copy()
    df["log_goal"] = np.log1p(df["card.targetAmount.value"].clip(lower=0))
    df["log_text_length"] = np.log1p(df["description_len_chars"].fillna(0))
    df["has_video"] = (df["video_count"].fillna(0) > 0).astype(int)
    return df


def build_X(df: pd.DataFrame, features: list) -> pd.DataFrame:
    cols = [c for c in features if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    # убираем константные
    X = X.loc[:, X.nunique() > 1]
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 0. Train / Test split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    min_class = y.value_counts().min()
    # при очень маленькой выборке стратификация невозможна — fallback
    stratify = y if min_class >= 2 and len(y) * test_size >= 2 else None
    if stratify is None:
        print("  ! маленькая выборка: стратификация отключена")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )
    return X_tr, X_te, y_tr, y_te


def normalize_readability_after_split(X_train, X_test):
    """
    Нормализует readability_avg и три индекса через z-score.
    Fit на train, transform на train и test.

    Избегает утечки данных: test не видит train statistics.
    Сохраняет параметры для применения к новым данным.

    Returns: X_train, X_test, scaler
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from normalization_scaler import ReadabilityScaler

    scaler = ReadabilityScaler(fit_on_train=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Сохранить параметры для воспроизводимости
    params_path = os.path.join(FIG_DIR, "readability_scaler_params.json")
    scaler.save(params_path)

    return X_train, X_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 1–2. Обучение + метрики
# ─────────────────────────────────────────────────────────────────────────────

def train_models(X_tr, y_tr):
    from sklearn.ensemble import RandomForestClassifier
    models = {}

    rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                class_weight="balanced", random_state=42)
    rf.fit(X_tr, y_tr)
    models["RandomForest"] = rf

    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              class_weight="balanced", random_state=42, verbose=-1)
        lgbm.fit(X_tr, y_tr)
        models["LightGBM"] = lgbm
    except ImportError:
        pass

    return models


def eval_metrics(models: dict, X_te, y_te, lines):
    from sklearn.metrics import (roc_auc_score, f1_score, average_precision_score,
                                 classification_report)
    log_print(lines, "\n── Метрики на test ──")
    rows = []
    for name, m in models.items():
        proba = m.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        row = {"model": name}
        row["ROC_AUC"] = roc_auc_score(y_te, proba) if y_te.nunique() > 1 else np.nan
        row["PR_AUC"] = average_precision_score(y_te, proba) if y_te.nunique() > 1 else np.nan
        row["F1"] = f1_score(y_te, pred, zero_division=0)
        rows.append(row)
        log_print(lines, f"\n{name}:")
        log_print(lines, classification_report(y_te, pred, zero_division=0))
    log_print(lines, pd.DataFrame(rows).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Permutation Importance
# ─────────────────────────────────────────────────────────────────────────────

def permutation_importance_plot(models: dict, X_te, y_te, lines):
    from sklearn.inspection import permutation_importance
    log_print(lines, "\n── Permutation Importance (test) ──")

    for name, model in models.items():
        result = permutation_importance(
            model, X_te, y_te,
            n_repeats=20, random_state=42, scoring="roc_auc"
        )
        imp = pd.Series(result.importances_mean, index=X_te.columns)
        imp_std = pd.Series(result.importances_std, index=X_te.columns)
        imp_sorted = imp.sort_values(ascending=False)
        top = imp_sorted.head(min(TOP_N * 2, len(imp_sorted)))

        fig, ax = plt.subplots(figsize=(8, max(5, len(top) * 0.35)))
        colors = ["#d73027" if v > 0 else "#4575b4" for v in top]
        ax.barh(top.index[::-1], top.values[::-1],
                xerr=imp_std.reindex(top.index[::-1]).values,
                color=colors[::-1], capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Permutation Importance — {name}\n(ΔROC-AUC при перемешивании признака)")
        ax.set_xlabel("ΔROC-AUC")
        plt.tight_layout()
        savefig(f"perm_imp_{name.lower()}")

        log_print(lines, f"\n{name} — топ признаки:")
        log_print(lines, imp_sorted.head(TOP_N).round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 4. Partial Dependence Plots + ICE
# ─────────────────────────────────────────────────────────────────────────────

def pdp_ice_plots(models: dict, X_tr, X_te, lines):
    from sklearn.inspection import PartialDependenceDisplay
    log_print(lines, "\n── PDP + ICE ──")

    # Выбираем топ признаки по feature_importances_ (RF)
    rf = models.get("RandomForest")
    if rf is None:
        return
    imp = pd.Series(rf.feature_importances_, index=X_tr.columns)
    top_features = imp.nlargest(min(TOP_N, len(imp))).index.tolist()
    log_print(lines, f"  признаки для PDP/ICE: {top_features}")

    # PDP (средний эффект)
    try:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.ravel()
        for i, feat in enumerate(top_features[:6]):
            PartialDependenceDisplay.from_estimator(
                rf, X_tr, [feat],
                kind="average",
                ax=axes[i],
                line_kw={"color": "#d73027"},
            )
            axes[i].set_title(feat, fontsize=9)
        for j in range(len(top_features), 6):
            axes[j].set_visible(False)
        fig.suptitle("Partial Dependence Plots (PDP) — RandomForest", fontsize=12)
        plt.tight_layout()
        savefig("pdp")
    except Exception as e:
        log_print(lines, f"  PDP ошибка: {e}")

    # ICE (индивидуальные траектории)
    try:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.ravel()
        for i, feat in enumerate(top_features[:6]):
            PartialDependenceDisplay.from_estimator(
                rf, X_tr, [feat],
                kind="both",          # ICE + PDP поверх
                ax=axes[i],
                ice_lines_kw={"color": "#4575b4", "alpha": 0.3, "linewidth": 0.8},
                pd_line_kw={"color": "#d73027", "linewidth": 2},
            )
            axes[i].set_title(feat, fontsize=9)
        for j in range(len(top_features), 6):
            axes[j].set_visible(False)
        fig.suptitle("ICE + PDP — RandomForest (синие = индивидуальные, красная = среднее)",
                     fontsize=11)
        plt.tight_layout()
        savefig("ice")
    except Exception as e:
        log_print(lines, f"  ICE ошибка: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. TreeSHAP: beeswarm, bar, dependence
# ─────────────────────────────────────────────────────────────────────────────

def treeshap_plots(models: dict, X_tr, X_te, lines):
    import shap
    log_print(lines, "\n── TreeSHAP ──")

    for name, model in models.items():
        log_print(lines, f"\n  {name}...")
        try:
            explainer = shap.TreeExplainer(model, data=X_tr)
            sv = explainer.shap_values(X_te)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
        except Exception as e:
            log_print(lines, f"    SHAP ошибка: {e}")
            continue

        feat_names = X_te.columns.tolist()
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feat_names)
        top_idx = mean_abs.nlargest(min(TOP_N * 2, len(feat_names))).index
        idx_nums = [feat_names.index(f) for f in top_idx]

        # Beeswarm
        plt.figure(figsize=(9, max(5, len(top_idx) * 0.4)))
        shap.summary_plot(sv[:, idx_nums],
                          X_te[top_idx.tolist()],
                          feature_names=top_idx.tolist(),
                          show=False, plot_size=None)
        plt.title(f"TreeSHAP beeswarm — {name}")
        plt.tight_layout()
        savefig(f"treeshap_beeswarm_{name.lower()}")

        # Bar
        plt.figure(figsize=(8, max(5, len(top_idx) * 0.35)))
        shap.summary_plot(sv[:, idx_nums],
                          X_te[top_idx.tolist()],
                          feature_names=top_idx.tolist(),
                          plot_type="bar", show=False, plot_size=None)
        plt.title(f"TreeSHAP bar — {name}")
        plt.tight_layout()
        savefig(f"treeshap_bar_{name.lower()}")

        # Dependence plots для топ-4 признаков
        top4 = mean_abs.nlargest(min(4, len(feat_names))).index.tolist()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        for i, feat in enumerate(top4):
            fidx = feat_names.index(feat)
            shap.dependence_plot(
                fidx, sv, X_te.values,
                feature_names=feat_names,
                ax=axes[i], show=False,
                dot_size=60,
            )
            axes[i].set_title(feat, fontsize=9)
        for j in range(len(top4), 4):
            axes[j].set_visible(False)
        fig.suptitle(f"SHAP Dependence Plots — {name}", fontsize=12)
        plt.tight_layout()
        savefig(f"treeshap_dependence_{name.lower()}")

        log_print(lines, f"    топ SHAP mean(|v|):")
        log_print(lines, mean_abs.nlargest(TOP_N).round(5).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Mean SHAP по категориям
# ─────────────────────────────────────────────────────────────────────────────

def shap_by_category(models: dict, X_te, df_te: pd.DataFrame, lines):
    import shap
    log_print(lines, "\n── Средний SHAP по категориям ──")

    if CAT_COL not in df_te.columns:
        log_print(lines, f"  колонка '{CAT_COL}' не найдена, пропускаю")
        return

    cats = df_te[CAT_COL].fillna("Неизвестно").values

    for name, model in models.items():
        try:
            explainer = shap.TreeExplainer(model, data=X_te)
            sv = explainer.shap_values(X_te)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
        except Exception as e:
            log_print(lines, f"  {name} ошибка: {e}")
            continue

        feat_names = X_te.columns.tolist()
        mean_abs_global = pd.Series(np.abs(sv).mean(axis=0), index=feat_names)
        top_feats = mean_abs_global.nlargest(min(8, len(feat_names))).index.tolist()

        shap_df = pd.DataFrame(sv, columns=feat_names, index=X_te.index)
        shap_df[CAT_COL] = cats

        grouped = shap_df.groupby(CAT_COL)[top_feats].mean()
        log_print(lines, f"\n{name} — средний SHAP по категориям (топ {len(top_feats)} признаков):")
        log_print(lines, grouped.round(4).to_string())

        if len(grouped) >= 2:
            fig, ax = plt.subplots(figsize=(max(8, len(top_feats)), max(4, len(grouped) * 0.6)))
            im = ax.imshow(grouped.values, cmap="RdBu_r", aspect="auto",
                           vmin=-grouped.abs().max().max(),
                           vmax=grouped.abs().max().max())
            ax.set_xticks(range(len(top_feats)))
            ax.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(grouped)))
            ax.set_yticklabels(grouped.index, fontsize=9)
            plt.colorbar(im, ax=ax, label="Средний SHAP")
            ax.set_title(f"Средний SHAP по категориям — {name}\n"
                         f"(красный = ↑ вероятность успеха, синий = ↓)")
            plt.tight_layout()
            savefig(f"shap_by_category_{name.lower()}")


# ─────────────────────────────────────────────────────────────────────────────
# 3-NEW. Визуализация числовых признаков
# ─────────────────────────────────────────────────────────────────────────────

def plot_numeric_features(X: pd.DataFrame, y: pd.Series, lines):
    """Гистограммы, корреляционная матрица, boxplot success vs fail."""
    log_print(lines, "\n── Визуализация числовых признаков ──")
    feats = X.columns.tolist()
    n = len(feats)

    # 1. Гистограммы по каждому признаку, раскрашены по классу
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = axes.ravel()
    colors = {0: "#4575b4", 1: "#d73027"}
    labels = {0: "fail", 1: "success"}
    for i, feat in enumerate(feats):
        ax = axes[i]
        for cls in sorted(y.unique()):
            vals = X.loc[y == cls, feat].dropna()
            ax.hist(vals, bins=15, alpha=0.6, color=colors[cls],
                    label=labels[cls], density=True)
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Распределение признаков: success (красный) vs fail (синий)", fontsize=11)
    plt.tight_layout()
    savefig("numeric_histograms")

    # 2. Корреляционная матрица (Spearman — устойчива к выбросам)
    corr = X.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(7, n * 0.5)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(feats, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(feats, fontsize=7)
    for r in range(n):
        for c in range(n):
            ax.text(c, r, f"{corr.values[r, c]:.2f}",
                    ha="center", va="center", fontsize=5,
                    color="white" if abs(corr.values[r, c]) > 0.6 else "black")
    plt.colorbar(im, ax=ax)
    ax.set_title("Матрица корреляций Спирмена", fontsize=11)
    plt.tight_layout()
    savefig("numeric_corr_matrix")

    # 3. Boxplot каждого признака success vs fail
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = axes.ravel()
    df_plot = X.copy()
    df_plot["_target"] = y.values
    for i, feat in enumerate(feats):
        ax = axes[i]
        data = [df_plot.loc[df_plot["_target"] == cls, feat].dropna().values
                for cls in [0, 1]]
        bp = ax.boxplot(data, labels=["fail", "success"], patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2})
        bp["boxes"][0].set_facecolor("#4575b4")
        bp["boxes"][1].set_facecolor("#d73027")
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Boxplot признаков: fail (синий) vs success (красный)", fontsize=11)
    plt.tight_layout()
    savefig("numeric_boxplots")

    # 4. Корреляция каждого признака с таргетом
    corr_target = X.apply(lambda col: col.corr(y, method="spearman")).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(5, len(corr_target) * 0.35)))
    colors_bar = ["#d73027" if v > 0 else "#4575b4" for v in corr_target]
    ax.barh(corr_target.index, corr_target.values, color=colors_bar)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Spearman-корреляция признаков с is_successful\n"
                 "(красный = положительная, синий = отрицательная)")
    ax.set_xlabel("r_spearman")
    plt.tight_layout()
    savefig("numeric_target_corr")

    log_print(lines, "  Spearman r с is_successful:")
    log_print(lines, corr_target.round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 9. GAM (pyGAM LogisticGAM)
# ─────────────────────────────────────────────────────────────────────────────

def gam_analysis(X_tr: pd.DataFrame, y_tr: pd.Series,
                 X_te: pd.DataFrame, y_te: pd.Series, lines):
    """LogisticGAM: partial effect каждого признака + метрики."""
    try:
        from pygam import LogisticGAM, s, f
        from sklearn.metrics import roc_auc_score
    except ImportError:
        log_print(lines, "\n(GAM пропущен — pip install pygam)")
        return

    log_print(lines, "\n── GAM (pyGAM LogisticGAM) ──")
    feat_names = X_tr.columns.tolist()
    n_feats = len(feat_names)

    # Строим формулу: все признаки — сплайны s()
    terms = s(0)
    for i in range(1, n_feats):
        terms = terms + s(i)

    gam = LogisticGAM(terms, max_iter=100)
    try:
        gam.fit(X_tr.values, y_tr.values)
    except Exception as e:
        log_print(lines, f"  GAM fit ошибка: {e}")
        return

    proba_te = gam.predict_proba(X_te.values)
    try:
        auc = roc_auc_score(y_te, proba_te)
        log_print(lines, f"  GAM ROC-AUC (test): {auc:.4f}")
    except Exception:
        pass

    # Partial effects — один признак за раз
    ncols = 4
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = axes.ravel()

    for i, feat in enumerate(feat_names):
        ax = axes[i]
        try:
            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
            ax.plot(XX[:, i], pdep, color="#d73027", linewidth=2)
            ax.fill_between(XX[:, i], confi[:, 0], confi[:, 1],
                            alpha=0.2, color="#d73027")
            # rug — фактические значения
            ax.scatter(X_tr[feat], np.zeros(len(X_tr)) - 0.05,
                       c=y_tr.map({0: "#4575b4", 1: "#d73027"}),
                       s=20, alpha=0.7, zorder=5)
        except Exception:
            pass
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(labelsize=6)

    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("GAM partial effects (95% CI)\n"
                 "rug: синий=fail, красный=success", fontsize=11)
    plt.tight_layout()
    savefig("gam_partial_effects")
    log_print(lines, "  ✓ GAM partial effects сохранены")

    # Сводка pseudo-R² и p-values
    try:
        log_print(lines, f"  pseudo-R²: {gam.statistics_['pseudo_r2']['mcfadden']:.4f}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 10. Learning Curves
# ─────────────────────────────────────────────────────────────────────────────

def learning_curves(models: dict, X: pd.DataFrame, y: pd.Series, lines):
    """ROC-AUC на train и val при разном размере обучающей выборки."""
    from sklearn.model_selection import learning_curve
    log_print(lines, "\n── Learning Curves ──")

    n = len(X)
    cv_splits = min(3, y.value_counts().min()) if y.value_counts().min() >= 2 else 2
    # train_sizes: дробные (0,1] — sklearn интерпретирует как доли
    # это надёжнее абсолютных значений при любом N
    n_points = min(8, max(3, n - cv_splits))
    train_sizes = np.linspace(0.2, 1.0, n_points)

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        try:
            tr_sizes, tr_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=cv_splits,
                scoring="roc_auc",
                n_jobs=-1,
                error_score=np.nan,
            )
        except Exception as e:
            log_print(lines, f"  {name}: ошибка learning_curve — {e}")
            ax.set_title(f"{name}\n(ошибка)")
            continue

        tr_mean = np.nanmean(tr_scores, axis=1)
        tr_std = np.nanstd(tr_scores, axis=1)
        val_mean = np.nanmean(val_scores, axis=1)
        val_std = np.nanstd(val_scores, axis=1)

        ax.plot(tr_sizes, tr_mean, "o-", color="#d73027", label="Train AUC")
        ax.fill_between(tr_sizes, tr_mean - tr_std, tr_mean + tr_std,
                        alpha=0.15, color="#d73027")
        ax.plot(tr_sizes, val_mean, "s-", color="#4575b4", label="Val AUC")
        ax.fill_between(tr_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color="#4575b4")
        ax.set_title(f"Learning Curve — {name}")
        ax.set_xlabel("Размер обучающей выборки")
        ax.set_ylabel("ROC-AUC")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

        log_print(lines, f"\n  {name}:")
        log_print(lines, f"    train_sizes: {tr_sizes.tolist()}")
        log_print(lines, f"    val AUC:     {np.round(val_mean, 3).tolist()}")

    plt.tight_layout()
    savefig("learning_curves")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Ошибки по сегментам
# ─────────────────────────────────────────────────────────────────────────────

def error_by_segments(models: dict, X_te: pd.DataFrame, y_te: pd.Series,
                      df_te_meta: pd.DataFrame, lines):
    """Ошибки (FP, FN, accuracy) по сегментам: категория, квартиль цели, has_video."""
    log_print(lines, "\n── Ошибки по сегментам ──")

    # Определяем сегменты из мета-данных test-части
    segment_cols = {}
    if CAT_COL in df_te_meta.columns:
        segment_cols["категория"] = df_te_meta[CAT_COL].fillna("Неизвестно")

    if "card.targetAmount.value" in df_te_meta.columns:
        target_amt = df_te_meta["card.targetAmount.value"].fillna(0)
        if target_amt.nunique() > 1:
            try:
                segment_cols["квартиль_цели"] = pd.qcut(
                    target_amt, q=min(4, target_amt.nunique()),
                    labels=False, duplicates="drop"
                ).astype(str)
            except Exception:
                pass

    if "has_video" in df_te_meta.columns:
        segment_cols["has_video"] = df_te_meta["has_video"].fillna(0).astype(int).astype(str)
    elif "has_video" in X_te.columns:
        segment_cols["has_video"] = X_te["has_video"].astype(int).astype(str)

    if not segment_cols:
        log_print(lines, "  нет подходящих колонок для сегментации")
        return

    for model_name, model in models.items():
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        error = (pred != y_te.values).astype(int)
        fp = ((pred == 1) & (y_te.values == 0)).astype(int)
        fn = ((pred == 0) & (y_te.values == 1)).astype(int)

        base_df = pd.DataFrame({
            "y_true": y_te.values,
            "y_pred": pred,
            "error": error,
            "FP": fp,
            "FN": fn,
            "proba": proba,
        }, index=X_te.index)

        n_seg = len(segment_cols)
        fig, axes = plt.subplots(1, n_seg, figsize=(5 * n_seg, 5))
        if n_seg == 1:
            axes = [axes]

        for ax, (seg_name, seg_series) in zip(axes, segment_cols.items()):
            seg_aligned = seg_series.reindex(X_te.index).fillna("Неизвестно")
            base_df["_seg"] = seg_aligned.values

            stats = base_df.groupby("_seg").agg(
                n=("y_true", "count"),
                accuracy=("error", lambda x: 1 - x.mean()),
                fp_rate=("FP", "mean"),
                fn_rate=("FN", "mean"),
                mean_proba=("proba", "mean"),
            ).reset_index()

            log_print(lines, f"\n  {model_name} | сегмент: {seg_name}")
            log_print(lines, stats.round(3).to_string(index=False))

            x_pos = range(len(stats))
            width = 0.25
            ax.bar([p - width for p in x_pos], stats["accuracy"],
                   width, label="Accuracy", color="#2b8cbe")
            ax.bar(x_pos, 1 - stats["fp_rate"],
                   width, label="1-FP rate", color="#74c476")
            ax.bar([p + width for p in x_pos], 1 - stats["fn_rate"],
                   width, label="1-FN rate", color="#fd8d3c")
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(stats["_seg"].astype(str), rotation=20,
                               ha="right", fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Доля")
            ax.set_title(f"{seg_name}\n(n={stats['n'].tolist()})", fontsize=9)
            ax.legend(fontsize=7)

        fig.suptitle(f"Ошибки по сегментам — {model_name}", fontsize=11)
        plt.tight_layout()
        savefig(f"errors_by_segment_{model_name.lower()}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. LIME
# ─────────────────────────────────────────────────────────────────────────────

def lime_explanations(models: dict, X_tr, X_te, y_te, lines):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        log_print(lines, "\nLIME не установлен (pip install lime)")
        return

    log_print(lines, f"\n── LIME: {LIME_N_EXAMPLES} примера из test ──")
    feat_names = X_tr.columns.tolist()

    explainer = LimeTabularExplainer(
        training_data=X_tr.values,
        feature_names=feat_names,
        class_names=["fail", "success"],
        mode="classification",
        random_state=42,
    )

    for name, model in models.items():
        predict_fn = lambda x: model.predict_proba(x)  # noqa: E731
        n = min(LIME_N_EXAMPLES, len(X_te))

        for i in range(n):
            row = X_te.iloc[i].values
            true_label = int(y_te.iloc[i])
            pred_prob = model.predict_proba(row.reshape(1, -1))[0, 1]

            exp = explainer.explain_instance(
                row, predict_fn, num_features=10, num_samples=200
            )
            fig = exp.as_pyplot_figure()
            fig.suptitle(
                f"LIME — {name} | пример #{i} | "
                f"true={'success' if true_label else 'fail'} | "
                f"P(success)={pred_prob:.2f}",
                fontsize=9
            )
            plt.tight_layout()
            savefig(f"lime_{name.lower()}_example{i}")

            top_exp = exp.as_list()
            log_print(lines, f"\n  {name}, пример #{i} "
                             f"(true={'success' if true_label else 'fail'}, "
                             f"P={pred_prob:.2f}):")
            for feat_str, val in top_exp[:6]:
                direction = "↑" if val > 0 else "↓"
                log_print(lines, f"    {direction} {feat_str}: {val:+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    lines = []

    def log(msg=""):
        log_print(lines, msg)

    log(f"→ читаю {INPUT}")
    df_raw = pd.read_excel(INPUT)
    df = prepare(df_raw)
    log(f"  shape: {df.shape}")

    all_features = CONTROLS + TEXT_FEATURES
    X = build_X(df, all_features)
    y = df[TARGET].astype(int)

    n_feat = X.shape[1]
    log(f"  признаков: {n_feat}, целевых: {y.sum()} / {len(y)}")
    log(f"  признаки: {X.columns.tolist()}")

    # ── 0. Split ─────────────────────────────────────────────────────────────
    log("\n── 0. Train / Test split (80/20) ──")
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    log(f"  train: {len(X_tr)}  test: {len(X_te)}")
    log(f"  train success/fail: {y_tr.sum()}/{(y_tr==0).sum()}")
    log(f"  test  success/fail: {y_te.sum()}/{(y_te==0).sum()}")

    # ── Нормализация readability признаков (ПОСЛЕ split!) ────────────────────
    log("\n── нормализация readability признаков (fit на train) ──")
    X_tr, X_te, scaler = normalize_readability_after_split(X_tr, X_te)
    log(f"  параметры сохранены: {FIG_DIR}/readability_scaler_params.json")
    log(scaler.summary())

    # ── 1. Обучение ───────────────────────────────────────────────────────────
    log("\n── 1. Обучение моделей ──")
    models = train_models(X_tr, y_tr)
    log(f"  обучено: {list(models.keys())}")

    # ── 2. Метрики ────────────────────────────────────────────────────────────
    eval_metrics(models, X_te, y_te, lines)

    # ── 3. Визуализация числовых признаков ──────────────────────────────────
    plot_numeric_features(X, y, lines)

    # ── 4. Permutation Importance ────────────────────────────────────────────
    permutation_importance_plot(models, X_te, y_te, lines)

    # ── 5. PDP + ICE ─────────────────────────────────────────────────────────
    pdp_ice_plots(models, X_tr, X_te, lines)

    # ── 6. TreeSHAP ──────────────────────────────────────────────────────────
    treeshap_plots(models, X_tr, X_te, lines)

    # ── 7. Mean SHAP по категориям ───────────────────────────────────────────
    df_te_meta = df.loc[X_te.index]
    shap_by_category(models, X_te, df_te_meta, lines)

    # ── 8. LIME ───────────────────────────────────────────────────────────────
    lime_explanations(models, X_tr, X_te, y_te, lines)

    # ── 9. GAM ───────────────────────────────────────────────────────────────
    gam_analysis(X_tr, y_tr, X_te, y_te, lines)

    # ── 10. Learning Curves ───────────────────────────────────────────────────
    learning_curves(models, X, y, lines)

    # ── 11. Ошибки по сегментам ──────────────────────────────────────────────
    error_by_segments(models, X_te, y_te, df_te_meta, lines)

    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    log(f"\n✓ отчёт: {REPORT}")


if __name__ == "__main__":
    main()
