"""
econometrics.py — Часть 3 плана text_analysis_plan.md.

  3.1 Logit (is_successful) с HC1-ошибками + OR, McFadden R², AIC/BIC, VIF
  3.2 OLS / Tobit (funding_pct или log_funding_pct) с HC1
  3.3 Поэтапная спецификация M0 → M1 → M2 → M3 → M_full → M_lasso

Вход: projects_sample_with_text.xlsx
Выход: econometrics_report.txt

Опции:
  python3 econometrics.py                           # OLS на funding_pct (по умолчанию)
  python3 econometrics.py --log-funding-pct        # OLS на log(1 + funding_pct)
  python3 econometrics.py --use-tobit              # Tobit регрессия (цензурировано слева на 0)
  python3 econometrics.py --use-tobit --log-funding-pct  # Tobit на log_funding_pct

Выбор метода:
  - --use-tobit (рекомендуется): проекты с funding_pct=0 обрабатываются корректно как цензурированные
  - --log-funding-pct: сжимает экстремальные значения, стабилизирует дисперсию
  - ничего: обычный OLS (может быть неправильным если много нулей)
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "projects_sample_with_text.xlsx")
REPORT = os.path.join(HERE, "econometrics_report.txt")

CONTROLS_BASE = [
    "log_goal", "campaign_duration_days",
    "counts.newsCount", "counts.commentsCount",
    "card.author.campaignsAmount",
    "has_video", "log_text_length",
]

TEXT_READ = ["readability_avg", "description_word_count"]
TEXT_SENT = ["rubert_positive", "rubert_negative"]
TEXT_LIWC = ["social_score", "gratitude_score", "we_ratio", "i_ratio",
             "certainty_score", "uncertainty_score"]


def prepare_data(df: pd.DataFrame, log_funding_pct: bool = False) -> pd.DataFrame:
    """
    Подготавливает датасет для эконометрики.

    Параметры:
      log_funding_pct (bool): если True, логарифмирует funding_pct через log1p.
                              Рекомендуется если распределение сильно скошено (много 0, outliers).
    """
    df = df.copy()
    df["log_goal"] = np.log1p(df["card.targetAmount.value"].clip(lower=0))
    df["log_text_length"] = np.log1p(df["description_len_chars"].fillna(0))
    df["has_video"] = (df["video_count"].fillna(0) > 0).astype(int)
    df["funding_pct"] = df["funding_ratio"].clip(lower=0)

    if log_funding_pct:
        df["log_funding_pct"] = np.log1p(df["funding_pct"])

    # категориальные дамми
    if "category_grouped" in df.columns:
        cat_d = pd.get_dummies(df["category_grouped"], prefix="cat", drop_first=True)
        df = pd.concat([df, cat_d], axis=1)
        df.attrs["cat_cols"] = cat_d.columns.tolist()
    else:
        df.attrs["cat_cols"] = []
    return df


def normalize_readability(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Z-score нормализует readability_avg и три индекса (flesch, fog, lix).
    Использует sample statistics (не требует train/test split для эконометрики).

    ВНИМАНИЕ: это работает для кросс-валидации в statsmodels, но если после этого
    будет делаться manual train/test, нужно переделать через fit на train!
    """
    from scipy.stats import zscore

    df = df.copy()
    readab_cols = [c for c in ["readability_flesch", "readability_fog", "readability_lix", "readability_avg"]
                   if c in df.columns and c in columns]

    for c in readab_cols:
        df[c] = zscore(df[c].dropna()).reindex(df.index)

    return df


def clean_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    features = [f for f in features if f in df.columns]
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.astype(float)
    # отбрасываем константные колонки
    X = X.loc[:, X.nunique() > 1]
    X = sm.add_constant(X, has_constant="add")
    return X


def fit_logit(X: pd.DataFrame, y: pd.Series, label: str, log) -> dict:
    log(f"\n── Logit: {label} ── (n={len(y)}, predictors={X.shape[1]-1})")
    if X.shape[1] >= len(y):
        log(f"  пропуск: предикторов ({X.shape[1]}) ≥ наблюдений ({len(y)})")
        return {"label": label, "converged": False, "skipped": True}
    try:
        model = sm.Logit(y, X).fit(disp=0, cov_type="HC1", maxiter=200)
    except Exception as e:
        log(f"  ОШИБКА: {e}")
        return {"label": label, "converged": False}
    log(model.summary().as_text())
    try:
        or_ci = np.exp(model.conf_int())
        or_ci.columns = ["OR_2.5%", "OR_97.5%"]
        or_ci["OR"] = np.exp(model.params)
        log("\nOdds Ratios (95% CI):")
        log(or_ci[["OR", "OR_2.5%", "OR_97.5%"]].round(4).to_string())
    except Exception:
        pass
    return {
        "label": label, "converged": True,
        "mcfadden_r2": getattr(model, "prsquared", np.nan),
        "aic": model.aic, "bic": model.bic, "llf": model.llf,
        "n_params": int(X.shape[1]),
    }


def fit_ols(X: pd.DataFrame, y: pd.Series, label: str, log) -> dict:
    log(f"\n── OLS: {label} ── (n={len(y)}, predictors={X.shape[1]-1})")
    if X.shape[1] >= len(y):
        log(f"  пропуск: предикторов ({X.shape[1]}) ≥ наблюдений ({len(y)})")
        return {"label": label, "r2": np.nan, "skipped": True}
    try:
        model = sm.OLS(y, X).fit(cov_type="HC1")
        log(model.summary().as_text())
    except Exception as e:
        log(f"  ОШИБКА: {e}")
        return {"label": label, "r2": np.nan}
    return {
        "label": label,
        "r2": getattr(model, "rsquared", np.nan),
        "adj_r2": getattr(model, "rsquared_adj", np.nan),
        "aic": getattr(model, "aic", np.nan),
        "bic": getattr(model, "bic", np.nan),
        "n_params": int(X.shape[1]),
    }


def compute_vif(X: pd.DataFrame, log) -> pd.DataFrame:
    cols = [c for c in X.columns if c != "const"]
    if len(cols) < 2:
        return pd.DataFrame()
    X_vif = X[cols].values
    vifs = []
    for i, c in enumerate(cols):
        try:
            v = variance_inflation_factor(X_vif, i)
        except Exception:
            v = np.nan
        vifs.append({"feature": c, "VIF": v})
    vif_df = pd.DataFrame(vifs).sort_values("VIF", ascending=False)
    log("\nVIF (порог > 5):")
    log(vif_df.round(3).to_string(index=False))
    return vif_df


def fit_tobit(X: pd.DataFrame, y: pd.Series, label: str, lower: float = 0.0, upper: float = None, log=None) -> dict:
    """
    Tobit регрессия (цензурированные данные).

    Параметры:
      X: матрица предикторов (с константой)
      y: целевая переменная (может быть цензурирована слева и/или справа)
      label: имя модели
      lower: нижняя граница цензурирования (default=0.0 для funding_pct)
      upper: верхняя граница цензурирования (default=None, нет)
      log: функция логирования

    Примечание: используем statsmodels.genmod.generalized_estimating_equations или
    собственную максимум-правдоподобную оценку через scipy.

    Tobit = Censored Regression:
      y_observed = max(lower, min(upper, y_latent))
      где y_latent = X·β + ε, ε ~ N(0, σ²)

    Интерпретация:
      - β коэффициенты показывают влияние на скрытую переменную y_latent
      - Маржинальный эффект на E[y_observed | y_observed > lower] меньше чем β
      - Используется когда много наблюдений на границе (в нашем случае funding_pct = 0)
    """
    if log is None:
        log = print

    try:
        from scipy.optimize import minimize
        from scipy.stats import norm
    except ImportError:
        log("  Tobit пропущен — нет scipy")
        return {"label": label, "converged": False, "method": "tobit"}

    log(f"\n── Tobit (censored regression): {label} ── (n={len(y)}, predictors={X.shape[1]-1}, lower={lower})")

    # Определить цензурированные наблюдения
    n_censored_lower = (y <= lower).sum()
    n_uncensored = (y > lower).sum()
    log(f"  цензурировано слева ({lower}): {n_censored_lower} / {len(y)}")
    log(f"  без цензуры: {n_uncensored} / {len(y)}")

    if n_uncensored < X.shape[1]:
        log(f"  пропуск: недостаточно uncensored наблюдений ({n_uncensored} < {X.shape[1]})")
        return {"label": label, "converged": False, "method": "tobit", "skipped": True}

    # Максимум-правдоподобная оценка (MLE)
    def neg_loglik(params):
        """Отрицательное логарифмическое правдоподобие."""
        beta = params[:-1]
        sigma = np.exp(params[-1])  # log-parametrization для положительности

        y_pred = X @ beta  # X·β
        u = (y - y_pred) / sigma  # стандартизованные остатки

        # Для цензурированных наблюдений (y <= lower):
        # P(y_observed = lower) = P(y_latent <= lower) = Φ((lower - X·β) / σ)
        censored_mask = y <= lower
        ll_censored = norm.logcdf(u[censored_mask]).sum() if censored_mask.sum() > 0 else 0

        # Для uncensored (y > lower):
        # f(y | y > lower) = φ((y - X·β) / σ) / (1 - Φ((lower - X·β) / σ))
        uncensored_mask = ~censored_mask
        if uncensored_mask.sum() > 0:
            logpdf = norm.logpdf(u[uncensored_mask]).sum()
            ll_uncensored = logpdf - np.log(sigma) * uncensored_mask.sum()
        else:
            ll_uncensored = 0

        total_ll = ll_censored + ll_uncensored
        return -total_ll  # минимизируем отрицательное логдоходимость

    # Начальные значения (от OLS на uncensored)
    uncensored_mask = y > lower
    if uncensored_mask.sum() > X.shape[1]:
        X_unc = X[uncensored_mask]
        y_unc = y[uncensored_mask]
        beta_init = np.linalg.lstsq(X_unc, y_unc, rcond=None)[0]
        sigma_init = np.std(y_unc - X_unc @ beta_init)
    else:
        beta_init = np.zeros(X.shape[1])
        sigma_init = np.std(y - y.mean())

    params_init = np.concatenate([beta_init, [np.log(max(sigma_init, 0.1))]])

    # Оптимизация
    result = minimize(neg_loglik, params_init, method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-6})

    if not result.success:
        log(f"  ВНИМАНИЕ: оптимизация не сошлась ({result.message})")

    beta = result.x[:-1]
    sigma = np.exp(result.x[-1])

    log(f"\nTobit Коэффициенты (эффект на скрытую y_latent = X·β):")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": beta,
    })
    log(coef_df.to_string(index=False))
    log(f"\nСигма (стд. ошибка): {sigma:.6f}")
    log(f"Log-likelihood: {-result.fun:.4f}")

    return {
        "label": label,
        "converged": result.success,
        "method": "tobit",
        "loglik": -result.fun,
        "sigma": sigma,
        "n_censored": n_censored_lower,
        "n_uncensored": n_uncensored,
    }


def lasso_logit(df: pd.DataFrame, features: list[str], y: pd.Series, log):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    features = [f for f in features if f in df.columns]
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    X = X.loc[:, X.nunique() > 1]
    if X.empty or y.nunique() < 2:
        log("  LASSO пропущен (недостаточно данных)")
        return
    Xs = StandardScaler().fit_transform(X)
    try:
        model = LogisticRegressionCV(Cs=10, penalty="l1", solver="saga",
                                     max_iter=5000, cv=min(3, y.value_counts().min()))
        model.fit(Xs, y)
    except Exception as e:
        log(f"  LASSO ОШИБКА: {e}")
        return
    coefs = pd.Series(model.coef_.ravel(), index=X.columns).sort_values(key=abs, ascending=False)
    nz = coefs[coefs != 0]
    log(f"\n── M_lasso: L1 logit (best C={model.C_[0]:.4f})")
    log(f"  отобрано признаков: {len(nz)} / {len(coefs)}")
    log(nz.round(4).to_string())


def main(log_funding_pct: bool = False, use_tobit: bool = False):
    """
    Параметры:
      log_funding_pct (bool): если True, OLS будет на log_funding_pct вместо funding_pct.
                              Используй если распределение funding_pct сильно скошено.
      use_tobit (bool): если True, использует Tobit регрессию вместо OLS.
                        Рекомендуется для funding_pct, так как много нулей (цензуировано слева).
    """
    lines: list[str] = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    log(f"→ читаю {INPUT}")
    df_raw = pd.read_excel(INPUT)
    df = prepare_data(df_raw, log_funding_pct=log_funding_pct)
    log(f"  shape: {df.shape}, успешных: {int(df['is_successful'].sum())} / {len(df)}")

    y_bin = df["is_successful"].astype(int)
    y_cont_col = "log_funding_pct" if log_funding_pct else "funding_pct"
    y_cont = df[y_cont_col].astype(float)

    if use_tobit:
        log(f"\n[INFO] Использую Tobit регрессию (цензурировано слева на 0)")
        log(f"       Целевая переменная: {y_cont_col} (цензурирована слева на funding_pct=0)")
    elif log_funding_pct:
        log(f"\n[INFO] OLS целевая переменная: {y_cont_col} (логарифмирован log1p)")
    else:
        log(f"\n[INFO] OLS целевая переменная: {y_cont_col} (оригинальный scale)")
    cat_cols = df.attrs["cat_cols"]

    # поэтапные спецификации
    specs = {
        "M0 (controls)": CONTROLS_BASE + cat_cols,
        "M1 (+readability)": CONTROLS_BASE + cat_cols + TEXT_READ,
        "M2 (+sentiment)": CONTROLS_BASE + cat_cols + TEXT_SENT,
        "M3 (+LIWC)": CONTROLS_BASE + cat_cols + TEXT_LIWC,
        "M_full": CONTROLS_BASE + cat_cols + TEXT_READ + TEXT_SENT + TEXT_LIWC,
    }

    log("\n" + "=" * 70)
    log("3.1 + 3.3 — LOGIT: поэтапная спецификация")
    log("=" * 70)
    logit_summary = []
    for name, feats in specs.items():
        X = clean_X(df, feats)
        logit_summary.append(fit_logit(X, y_bin, name, log))
    log("\n── Сравнение моделей (Logit) ──")
    log(pd.DataFrame(logit_summary).to_string(index=False))

    log("\n" + "=" * 70)
    if use_tobit:
        log("3.2 + 3.3 — Tobit регрессия (цензурировано слева на 0)")
    else:
        log("3.2 + 3.3 — OLS на funding_pct")
    log("=" * 70)

    if use_tobit:
        tobit_summary = []
        for name, feats in specs.items():
            X = clean_X(df, feats)
            tobit_summary.append(fit_tobit(X, y_cont, name, lower=0.0, log=log))
        log("\n── Сравнение моделей (Tobit) ──")
        log(pd.DataFrame(tobit_summary).to_string(index=False))
    else:
        ols_summary = []
        for name, feats in specs.items():
            X = clean_X(df, feats)
            ols_summary.append(fit_ols(X, y_cont, name, log))
        log("\n── Сравнение моделей (OLS) ──")
        log(pd.DataFrame(ols_summary).to_string(index=False))

    log("\n" + "=" * 70)
    log("VIF — мультиколлинеарность M_full")
    log("=" * 70)
    X_full = clean_X(df, specs["M_full"])
    compute_vif(X_full, log)

    log("\n" + "=" * 70)
    log("3.3 — M_lasso (L1-регуляризация, автоматический отбор)")
    log("=" * 70)
    lasso_logit(df, specs["M_full"], y_bin, log)

    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ отчёт: {REPORT}")


if __name__ == "__main__":
    import sys
    log_funding_pct = "--log-funding-pct" in sys.argv
    use_tobit = "--use-tobit" in sys.argv
    main(log_funding_pct=log_funding_pct, use_tobit=use_tobit)
