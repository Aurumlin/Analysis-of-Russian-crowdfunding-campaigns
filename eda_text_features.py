"""
eda_text_features.py — Часть 2 плана text_analysis_plan.md.

Предварительный анализ текстовых признаков:
  • описательная статистика по группам success / fail
  • Mann-Whitney U-test + effect size (rank-biserial r)
  • точечно-бисериальная корреляция с is_successful
  • violin-plots на каждый признак

Вход: projects_sample_with_text.xlsx (из extract_features.py)
Выход:
  • eda_results.csv       — сводная таблица тестов
  • figures/violin_*.png  — violin-plots
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "projects_sample_with_text.xlsx")
FIG_DIR = os.path.join(HERE, "figures")
OUT_CSV = os.path.join(HERE, "eda_results.csv")

TEXT_FEATURES = [
    "social_score",
    "gratitude_score", "has_gratitude",
    "we_count", "i_count", "we_ratio", "we_vs_i",
    "certainty_score", "uncertainty_score",
    "readability_flesch", "readability_fog", "readability_lix", "readability_avg",
    "rubert_positive", "rubert_negative", "rubert_neutral",
    "topic_0", "topic_1", "topic_2", "topic_3", "topic_4",
]
TARGET = "is_successful"


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Effect size для Mann-Whitney U: r = 1 − 2U / (n1·n2)."""
    return 1 - (2 * u_stat) / (n1 * n2)


def analyze_feature(df: pd.DataFrame, feature: str, target: str = TARGET) -> dict:
    success = df.loc[df[target] == 1, feature].dropna()
    fail = df.loc[df[target] == 0, feature].dropna()

    res = {
        "feature": feature,
        "n_success": len(success),
        "n_fail": len(fail),
        "median_success": success.median() if len(success) else np.nan,
        "median_fail": fail.median() if len(fail) else np.nan,
        "mean_success": success.mean() if len(success) else np.nan,
        "mean_fail": fail.mean() if len(fail) else np.nan,
    }

    if len(success) >= 2 and len(fail) >= 2 and success.nunique() + fail.nunique() > 2:
        u, p = stats.mannwhitneyu(success, fail, alternative="two-sided")
        res["mw_U"] = u
        res["mw_p"] = p
        res["rank_biserial_r"] = rank_biserial(u, len(success), len(fail))
    else:
        res["mw_U"] = np.nan
        res["mw_p"] = np.nan
        res["rank_biserial_r"] = np.nan

    combined = df[[feature, target]].dropna()
    if combined[feature].nunique() > 1 and combined[target].nunique() > 1:
        r_pb, p_pb = stats.pointbiserialr(combined[target], combined[feature])
        res["pointbiserial_r"] = r_pb
        res["pointbiserial_p"] = p_pb
    else:
        res["pointbiserial_r"] = np.nan
        res["pointbiserial_p"] = np.nan

    return res


def plot_violin(df: pd.DataFrame, feature: str, target: str = TARGET):
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df, x=target, y=feature, inner="quartile", cut=0)
    sns.stripplot(data=df, x=target, y=feature, color="black", alpha=0.5, size=3)
    plt.title(f"{feature} by {target}")
    plt.xlabel("is_successful (0=fail, 1=success)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"violin_{feature}.png"), dpi=120)
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"→ читаю {INPUT}")
    df = pd.read_excel(INPUT)
    print(f"  shape: {df.shape}")
    print(f"  {TARGET}: {df[TARGET].value_counts().to_dict()}")

    features = [f for f in TEXT_FEATURES if f in df.columns]
    print(f"→ анализирую {len(features)} признаков")

    rows = [analyze_feature(df, f) for f in features]
    results = pd.DataFrame(rows)

    results_sorted = results.sort_values("mw_p", na_position="last")
    display_cols = [
        "feature", "median_success", "median_fail",
        "mw_U", "mw_p", "rank_biserial_r",
        "pointbiserial_r", "pointbiserial_p",
    ]
    print("\n=== ТАБЛИЦА ТЕСТОВ ===")
    with pd.option_context("display.max_rows", None, "display.width", 160,
                           "display.float_format", "{:.4f}".format):
        print(results_sorted[display_cols].to_string(index=False))

    results_sorted.to_csv(OUT_CSV, index=False)
    print(f"\n✓ сохранено: {OUT_CSV}")

    print(f"→ рисую violin-plots в {FIG_DIR}/")
    for f in features:
        plot_violin(df, f)
    print(f"✓ готово: {len(features)} графиков")

    sig = results[results["mw_p"] < 0.1]
    if len(sig):
        print(f"\n≈ признаков с p<0.1: {len(sig)}")
        print(sig[["feature", "mw_p", "rank_biserial_r"]].to_string(index=False))
    else:
        print("\n(на выборке 10 проектов значимых различий не ожидается — "
              "скрипт будет показательнее на полном датасете)")


if __name__ == "__main__":
    main()
