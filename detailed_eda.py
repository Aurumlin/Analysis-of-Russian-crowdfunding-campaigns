"""
detailed_eda.py — Подробный разведочный анализ projects_sample_with_text.xlsx.

Запускается после extract_features.py и сохраняет результаты в папку `eda_detailed/`:

  eda_detailed/
    ├── summary/
    │   ├── overview.csv              — общая статистика по всем признакам
    │   ├── numeric_describe.csv      — describe() для числовых
    │   ├── categorical_counts.csv    — value_counts() для категориальных
    │   ├── missing_report.csv        — пропущенные значения
    │   └── target_balance.csv        — баланс is_successful
    ├── distributions/                — гистограмма + boxplot для каждой числовой
    ├── correlations/                 — heatmap матриц корреляций
    ├── target_analysis/              — violin / box по is_successful
    ├── bivariate/                    — scatter vs funding_ratio
    ├── categorical/                  — bar plots категорий
    └── REPORT.md                     — сводный отчёт (Markdown)

Запуск:  python3 detailed_eda.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 100

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "projects_sample_with_text.xlsx")
OUT_DIR = os.path.join(HERE, "eda_detailed")

TARGET_BIN = "is_successful"
TARGET_CONT = "funding_ratio"

# Категориальные / текстовые колонки, которые пропускаем в числовом анализе
SKIP_COLS = {
    "project_key", "sourceUrl", "card.title", "card.subtitle",
    "description.text", "clean_text", "meta.description",
    "card.startAt", "card.finishAt",
    "card.links.vk_url", "card.links.telegram_url", "card.links.author_site_url",
    "card.author.id",
}


def ensure_dirs():
    for sub in ("summary", "distributions", "correlations",
                "target_analysis", "bivariate", "categorical"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ОБЩАЯ СВОДКА
# ─────────────────────────────────────────────────────────────────────────────

def write_overview(df: pd.DataFrame) -> dict:
    print("→ сводка overview / missing / target balance")
    overview = pd.DataFrame({
        "dtype":       df.dtypes.astype(str),
        "n_unique":    df.nunique(dropna=True),
        "n_missing":   df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "sample":      df.iloc[0].astype(str).str.slice(0, 60),
    })
    overview.to_csv(os.path.join(OUT_DIR, "summary", "overview.csv"))

    numeric = df.select_dtypes(include=[np.number])
    numeric.describe().T.round(4).to_csv(
        os.path.join(OUT_DIR, "summary", "numeric_describe.csv")
    )

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_rows = []
    for c in cat_cols:
        if c in SKIP_COLS:
            continue
        vc = df[c].value_counts(dropna=False).head(20)
        for val, cnt in vc.items():
            cat_rows.append({"column": c, "value": str(val)[:80], "count": cnt})
    pd.DataFrame(cat_rows).to_csv(
        os.path.join(OUT_DIR, "summary", "categorical_counts.csv"), index=False
    )

    missing = (df.isna().sum() / len(df) * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)
    missing.to_frame("pct_missing").to_csv(
        os.path.join(OUT_DIR, "summary", "missing_report.csv")
    )

    target_balance = None
    if TARGET_BIN in df.columns:
        tb = df[TARGET_BIN].value_counts(dropna=False)
        target_balance = pd.DataFrame({
            "class": tb.index.astype(str),
            "count": tb.values,
            "pct": (tb.values / len(df) * 100).round(2),
        })
        target_balance.to_csv(
            os.path.join(OUT_DIR, "summary", "target_balance.csv"), index=False
        )

    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_numeric": numeric.shape[1],
        "n_categorical": len(cat_cols),
        "n_missing_total": int(df.isna().sum().sum()),
        "target_balance": target_balance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. РАСПРЕДЕЛЕНИЯ
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame):
    print("→ гистограммы + boxplot распределений")
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.drop(columns=[c for c in SKIP_COLS if c in num.columns], errors="ignore")
    num = num.loc[:, num.nunique() > 1]

    for col in num.columns:
        data = num[col].dropna()
        if data.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].hist(data, bins=min(30, max(5, len(data) // 2)), color="#4C72B0", edgecolor="black")
        axes[0].set_title(f"Histogram: {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("count")

        axes[1].boxplot(data, vert=False)
        axes[1].set_title(f"Boxplot: {col}")
        axes[1].set_xlabel(col)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "distributions",
                                 f"dist_{_safe_filename(col)}.png"))
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. КОРРЕЛЯЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlations(df: pd.DataFrame):
    print("→ корреляционные матрицы")
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.drop(columns=[c for c in SKIP_COLS if c in num.columns], errors="ignore")
    num = num.loc[:, num.nunique() > 1]

    if num.shape[1] < 2:
        return

    for method in ("pearson", "spearman"):
        corr = num.corr(method=method)
        corr.round(4).to_csv(
            os.path.join(OUT_DIR, "correlations", f"corr_{method}.csv")
        )

        fig, ax = plt.subplots(figsize=(max(8, num.shape[1] * 0.35),
                                        max(6, num.shape[1] * 0.35)))
        if HAS_SNS:
            sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,
                        square=True, cbar_kws={"shrink": 0.6}, ax=ax)
        else:
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns)
            plt.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title(f"Correlation matrix ({method})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "correlations",
                                 f"heatmap_{method}.png"))
        plt.close(fig)

    # Топ-корреляции с целевой
    top = []
    for target in (TARGET_BIN, TARGET_CONT):
        if target not in num.columns:
            continue
        corr_with = num.corr(method="spearman")[target].drop(target)
        corr_with = corr_with.abs().sort_values(ascending=False).head(20)
        top.append(pd.DataFrame({
            "feature": corr_with.index,
            "abs_corr_spearman": corr_with.values.round(4),
            "target": target,
        }))
    if top:
        pd.concat(top).to_csv(
            os.path.join(OUT_DIR, "correlations", "top_corr_with_target.csv"),
            index=False
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. АНАЛИЗ ПО ЦЕЛЕВОЙ
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_analysis(df: pd.DataFrame):
    if TARGET_BIN not in df.columns:
        return
    print(f"→ анализ признаков по {TARGET_BIN}")
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.drop(columns=[c for c in SKIP_COLS if c in num.columns], errors="ignore")
    num = num.loc[:, num.nunique() > 1]
    if TARGET_BIN not in num.columns:
        num[TARGET_BIN] = df[TARGET_BIN]

    group_stats = []
    for col in num.columns:
        if col == TARGET_BIN:
            continue
        grp = df.groupby(TARGET_BIN)[col].agg(["mean", "median", "std", "count"])
        grp["feature"] = col
        group_stats.append(grp.reset_index())

        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            if HAS_SNS:
                sns.violinplot(x=df[TARGET_BIN], y=df[col], ax=ax,
                               inner="box", cut=0)
            else:
                groups = [df[df[TARGET_BIN] == g][col].dropna()
                          for g in sorted(df[TARGET_BIN].dropna().unique())]
                ax.boxplot(groups, labels=sorted(df[TARGET_BIN].dropna().unique()))
        except Exception:
            plt.close(fig)
            continue
        ax.set_title(f"{col} by {TARGET_BIN}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "target_analysis",
                                 f"target_{_safe_filename(col)}.png"))
        plt.close(fig)

    if group_stats:
        all_stats = pd.concat(group_stats, ignore_index=True)
        all_stats.to_csv(
            os.path.join(OUT_DIR, "target_analysis", "group_stats.csv"),
            index=False
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. BIVARIATE (vs funding_ratio)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bivariate(df: pd.DataFrame):
    if TARGET_CONT not in df.columns:
        return
    print(f"→ scatter-plots vs {TARGET_CONT}")
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.drop(columns=[c for c in SKIP_COLS if c in num.columns], errors="ignore")
    num = num.loc[:, num.nunique() > 1]

    for col in num.columns:
        if col == TARGET_CONT:
            continue
        data = df[[col, TARGET_CONT]].dropna()
        if len(data) < 3:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(data[col], data[TARGET_CONT], alpha=0.6, color="#4C72B0")
        # линия тренда
        try:
            m, b = np.polyfit(data[col], data[TARGET_CONT], 1)
            xs = np.linspace(data[col].min(), data[col].max(), 50)
            ax.plot(xs, m * xs + b, color="red", linewidth=1.5,
                    label=f"y={m:.3f}x+{b:.3f}")
            ax.legend()
        except Exception:
            pass
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET_CONT)
        ax.set_title(f"{col} vs {TARGET_CONT}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "bivariate",
                                 f"scatter_{_safe_filename(col)}.png"))
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. КАТЕГОРИАЛЬНЫЕ
# ─────────────────────────────────────────────────────────────────────────────

def plot_categoricals(df: pd.DataFrame):
    print("→ bar-plots категориальных")
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if c not in SKIP_COLS and df[c].nunique() <= 30]

    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).head(20)
        if vc.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(vc) * 0.5), 4))
        ax.bar(range(len(vc)), vc.values, color="#55A868")
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels([str(x)[:20] for x in vc.index], rotation=45, ha="right")
        ax.set_title(f"Counts: {col}")
        ax.set_ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "categorical",
                                 f"cat_{_safe_filename(col)}.png"))
        plt.close(fig)

        # Success rate per category
        if TARGET_BIN in df.columns:
            sr = df.groupby(col)[TARGET_BIN].agg(["mean", "count"])
            sr = sr[sr["count"] >= 1].sort_values("mean", ascending=False).head(20)
            if not sr.empty:
                fig, ax = plt.subplots(figsize=(max(6, len(sr) * 0.5), 4))
                ax.bar(range(len(sr)), sr["mean"].values, color="#C44E52")
                ax.set_xticks(range(len(sr)))
                ax.set_xticklabels([str(x)[:20] for x in sr.index],
                                   rotation=45, ha="right")
                ax.set_title(f"Success rate by {col}")
                ax.set_ylabel(f"mean({TARGET_BIN})")
                ax.set_ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, "categorical",
                                         f"success_{_safe_filename(col)}.png"))
                plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7. СВОДНЫЙ ОТЧЁТ
# ─────────────────────────────────────────────────────────────────────────────

def write_report(meta: dict, df: pd.DataFrame):
    print("→ пишу REPORT.md")
    lines = []
    lines.append(f"# Detailed EDA — projects_sample_with_text.xlsx\n")
    lines.append(f"**Строк:** {meta['n_rows']}  \n")
    lines.append(f"**Колонок:** {meta['n_cols']}  \n")
    lines.append(f"**Числовых признаков:** {meta['n_numeric']}  \n")
    lines.append(f"**Категориальных:** {meta['n_categorical']}  \n")
    lines.append(f"**Всего пропущенных:** {meta['n_missing_total']}\n\n")

    if meta.get("target_balance") is not None:
        lines.append("## Баланс целевой переменной\n\n")
        lines.append(meta["target_balance"].to_markdown(index=False))
        lines.append("\n\n")

    lines.append("## Что где искать\n\n")
    lines.append("- **`summary/`** — общие таблицы (dtypes, describe, missing)\n")
    lines.append("- **`distributions/`** — гистограмма + boxplot каждой числовой переменной\n")
    lines.append("- **`correlations/`** — матрицы Пирсона/Спирмена + топ-корреляции с таргетом\n")
    lines.append(f"- **`target_analysis/`** — violin-plots в разрезе `{TARGET_BIN}` + `group_stats.csv`\n")
    lines.append(f"- **`bivariate/`** — scatter-plots каждого признака vs `{TARGET_CONT}`\n")
    lines.append("- **`categorical/`** — bar-plots категорий + success rate по категориям\n\n")

    # Топ корреляций
    top_path = os.path.join(OUT_DIR, "correlations", "top_corr_with_target.csv")
    if os.path.exists(top_path):
        top = pd.read_csv(top_path)
        for target in top["target"].unique():
            sub = top[top["target"] == target].head(10)
            lines.append(f"## Топ-10 корреляций с `{target}` (|Spearman|)\n\n")
            lines.append(sub[["feature", "abs_corr_spearman"]].to_markdown(index=False))
            lines.append("\n\n")

    # Список сгенерированных файлов
    lines.append("## Сгенерированные файлы\n\n")
    total = 0
    for sub in ("summary", "distributions", "correlations",
                "target_analysis", "bivariate", "categorical"):
        path = os.path.join(OUT_DIR, sub)
        if not os.path.exists(path):
            continue
        files = sorted(os.listdir(path))
        total += len(files)
        lines.append(f"- **{sub}/** — {len(files)} файлов\n")
    lines.append(f"\n**Итого:** {total} файлов\n")

    with open(os.path.join(OUT_DIR, "REPORT.md"), "w") as f:
        f.writelines(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(INPUT):
        print(f"✗ не найден {INPUT}. Сначала запусти extract_features.py.")
        return 1

    print(f"→ читаю {INPUT}")
    df = pd.read_excel(INPUT)
    print(f"  shape: {df.shape}")

    ensure_dirs()
    meta = write_overview(df)
    plot_distributions(df)
    plot_correlations(df)
    plot_target_analysis(df)
    plot_bivariate(df)
    plot_categoricals(df)
    write_report(meta, df)

    print(f"\n✓ EDA завершён — см. {OUT_DIR}/REPORT.md")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
