# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis research project analyzing success factors of reward-based crowdfunding campaigns on Russian platforms (Planeta.ru and Boomstarter). The goal is to identify which campaign characteristics (text, media, financial, temporal, author) statistically predict campaign success.

## Running the Analysis

### Full Pipeline (recommended)

```bash
# Run all steps: extract → eda → econ → ml → interp → tfidf
python3 run_all.py

# Only specific steps
python3 run_all.py --only extract,ml

# Skip heavy steps
python3 run_all.py --skip tfidf,interp

# With Tobit regression and log-transform
python3 run_all.py --use-tobit --log-funding-pct
```

### Parsing raw data

For initial JSON → DataFrame parsing, use the notebook:

```bash
jupyter lab Planeta.ipynb
```

The notebook cells must be executed sequentially. The main data pipeline expects a `planeta_parsed_csv/` folder with CSV input files.

## Key Files

- **`Planeta.ipynb`** — Main analysis notebook (41 cells). All feature engineering, EDA, and exports live here.
- **`Readme_diploma.md`** — Data dictionary: 73 Planeta columns and 45+ Boomstarter columns with field descriptions and usage notes (marks which columns are useful vs. redundant).
- **`Диплом.txt`** — Full thesis text in Russian (theoretical framework, literature review, methodology).
- **`projects_planeta.xlsx`** — Raw Planeta.ru dataset (~12 MB).
- **`projects_sample.xlsx`** — Smaller sample for quick iteration.

## Architecture

### Data Flow

```
Raw JSON/Excel (projects_planeta.xlsx)
  → parse_planeta_json_to_dfs()       # flattens nested JSON into DataFrames
  → normalize_planeta_dtypes()         # casts dates, numerics, categories
  → Feature Engineering                # text lengths, financial ratios, media counts
  → save_dfs_to_csv_folder() / save_dfs_to_excel()   # exports for modeling
```

### DataFrames Produced

| DataFrame | Contents |
|---|---|
| `projects_df` | Main project data (70+ fields) |
| `rewards_df` | Reward tier details (price, availability, delivery) |
| `donate_df` | Donation options |
| `media_images_df` / `media_videos_df` / `media_links_df` | Media references |
| `root_df` | Parser metadata |

### Key Functions in Notebook

- `parse_planeta_json_to_dfs(json_data)` — Entry point for raw JSON → DataFrames
- `flatten_dict(d, sep=".")` — Flattens nested dicts with dot-separated keys
- `normalize_planeta_dtypes()` / `cast_columns()` — Type normalization
- `group_category()` — Standardizes project category labels
- `save_dfs_to_csv_folder(dfs, path)` / `save_dfs_to_excel(dfs, filename)` — Exports

### Feature Categories

- **Text**: character/word counts for title, subtitle, description, meta-description
- **Financial**: funding ratio, collected vs. target amounts, success flag
- **Media**: image count, video count, external link count
- **Temporal**: campaign duration, start/finish dates, days remaining
- **Author**: creator project count, creator ID
- **Engagement**: comments, news posts, participants, purchases

## Text Feature Extraction Pipeline

**Script**: `extract_features.py` (run via `python3 extract_features.py`)

Extracts text-based features from `projects_sample.xlsx` using **separate text cleaning for each task**:

### Features Extracted

1. **social_score** — Words indicating social impact (благодаря, сообщество, etc.)
   - Formula: `social_count / len(lemmas)` — % of lemmatized words
   
2. **gratitude_score** — Regex patterns for gratitude (спасибо*, благодар*, etc.)
   - Formula: `matches / len(tokens)` — % of raw words
   
3. **collectivism** — `we_count`, `i_count`, `we_ratio`, `i_ratio` measure group vs. individual language
   - Formula: `we_count / len(tokens)` — % of "we" pronouns from all words
   - Formula: `i_count / len(tokens)` — % of "I" pronouns from all words
   - Formula: `we_vs_i = we_count / (i_count + 1)` — dimensionless ratio
   
4. **certainty** — `certainty_score` vs. `uncertainty_score` using vocabulary matching
   - Formula: `certainty_count / len(lemmas)` — % of lemmatized words
   
5. **numeric features** — Concreteness: `money_mentions`, `number_density`, `has_specific_sum`
   - `number_density` = % of words that are digits (e.g., "500", "10", "50,000")
   - `money_mentions` = count of currency references (e.g., "₽50,000", "2 млн")
   - `has_specific_sum` = binary flag (text mentions specific amount?)
   
6. **readability** — Flesch, Fog, LIX, and average readability indices
   - Preserves digits and punctuation (not removed)

7. **RuBERT sentiment** — Pre-trained transformer for positive/negative/neutral classification
   - Minimal cleaning (preserves punctuation, digits, capitalization)

8. **LDA topics** — Unsupervised topic modeling with auto K selection (default K=2–10 range, via Coherence Score)

### Separate Text Cleaning for Each Task

| Feature Group | Cleaning Function | Preserves | Use Case |
|---|---|---|---|
| 1.1–1.4 (LIWC) | `clean_text_for_liwc()` | Words only | Dictionary matching needs only words |
| 1.5 (Readability) | `clean_text_for_readability()` | All: punctuation + digits | Sentence/word length metrics need these |
| 1.6 (RuBERT) | `clean_text_for_rubert()` | Everything except HTML | Transformer models handle punctuation well |
| 1.7 (LDA) | `clean_text_for_lda()` | Words only | Topic modeling filters digits |

**Example:** Text "We printed 500 books for ₽50,000"

| Cleaning | Output | Why |
|---|---|---|
| `clean_text_for_liwc` | "We printed books for" | Only words for vocabulary matching |
| `clean_text_for_readability` | "We printed 500 books for ₽50,000" | Digits and punctuation affect readability metrics |
| `clean_text_for_rubert` | "We printed 500 books for ₽50,000" | Transformer handles everything natively |
| `clean_text_for_lda` | "We printed books for" | Only words for topic modeling |

### Unified Denominators (Problem 10)

All features are now **normalized to tokens (raw words)** or **lemmas (normalized forms)** consistently:

- **Lemma-based**: social_score, gratitude_score, certainty_score, uncertainty_score (normalized to # lemmas)
- **Token-based**: we_ratio, i_ratio (normalized to # raw words)

This ensures comparable, interpretable coefficients in regression: *"X% of the text exhibits property Y."*

See DOCS_TECHNICAL.md § 10 for detailed formulas and rationale.

### LDA Topic Selection (Coherence Score)

By default, `extract_features.py` automatically selects the optimal number of topics (K) using **Coherence Score** (`u_mass` metric from gensim). This avoids the arbitrary choice of K=5.

```
[LDA] подбор K по Coherence Score...
  K=2... score=0.4521
  K=3... score=0.5123 ← best
  K=4... score=0.4998
  K=5... score=0.5034
  ...
[LDA] ✓ лучший K=3 (score=0.5123)
```

**How it works:**
1. Tests K from 2–10 (configurable via `k_range` parameter)
2. Computes `u_mass` coherence for each K
3. Selects K with **highest coherence score**
4. If corpus is very small (N < 20), falls back to K=5 for stability

**Parameter tuning in `extract_features.py`:**
```python
# Auto-select K in range [2, 11) with u_mass coherence
df = add_lda_topics(df, auto_k=True, k_range=range(2, 11))

# Or use fixed K=5 for reproducibility
df = add_lda_topics(df, auto_k=False)  # K=5 hardcoded
```

This ensures your topics are data-driven and comparable across different corpus sizes.

### Dictionary Coverage Validation

After extraction, `extract_features.py` prints **coverage statistics** showing what % of texts contain words from each dictionary:

```
ВАЛИДАЦИЯ: Покрытие словарей
─────────────────────────────────────────
✓ social_score          :  73.5% (147/200 текстов)
⚠️  gratitude_score      :   8.2%  (16/200 текстов)
✓ certainty_score       :  45.0%  (90/200 текстов)
✓ uncertainty_score     :  62.5% (125/200 текстов)
─────────────────────────────────────────
```

**Interpretation**: If coverage < 10%, the dictionary is not effective on this corpus. Actions:
- Expand the dictionary with more words/patterns (edit `dicts.py`)
- Verify the detection logic in `text_features.py`
- Consider alternative approaches (e.g., semantic embeddings)

The dictionaries are editable in `dicts.py`; changes take effect on next run.

## Econometrics & Target Variable Modeling

**Script**: `econometrics.py`

Three approaches to handle `funding_pct`:

### 1. Tobit Regression (Recommended) 🎯

For censored data with many zeros (projects that funded 0% of their target):

```bash
python3 econometrics.py --use-tobit
```

**Best for:**
- Data with significant proportion of zeros (censoring point)
- Theoretically correct MLE approach for left-censored data
- Preserves information about extreme values

**How it works:**
- Explicitly models latent variable: `y_latent = X·β + ε, ε ~ N(0, σ²)`
- Observed: `y = max(0, y_latent)` (left-censored at 0)
- Coefficients show effect on latent (unobserved) variable

### 2. Log-Transform OLS

Compress skewed distribution without modeling censoring:

```bash
python3 econometrics.py --log-funding-pct
```

**Best for:**
- Extreme values that need compression
- Faster, simpler interpretation
- When censoring is not the primary concern

**Coefficient interpretation:**
If β₁ = 0.25 → funding_pct increases by ~28% (= exp(0.25) − 1)

### 3. Plain OLS (Default, Not Recommended)

```bash
python3 econometrics.py
```

**Issues:** Violates normality assumption, unstable errors with zero-heavy data.

### Comparison & Guidance

| Scenario | Use |
|---|---|
| Many zeros (>20%), want theoretical soundness | `--use-tobit` |
| Many zeros, want simplicity & speed | `--log-funding-pct` |
| Few zeros, roughly symmetric | default OLS |

**Combined:** `--use-tobit --log-funding-pct` applies Tobit to log-transformed funding_pct.

See DOCS_TECHNICAL.md § 8–9 for detailed diagnostic code.

## Machine Learning Models

**Script**: `ml_models.py`

Trains 5 classifiers to predict `is_successful` (binary target: did project reach goal?).

### Feature Sets

Three feature combinations are evaluated:

| Set | Contents | Use case |
|---|---|---|
| **A** | Controls + category dummies | Baseline: non-text predictors only |
| **B** | Text features only | Measure text-only predictive power |
| **C** | A + B (all features) | Full model: controls + category + text |

**Category dummies** (e.g., `cat_Творческие`, `cat_Образование`) are one-hot encoded from `category_grouped`. Baseline category is dropped to avoid multicollinearity.

### Models Evaluated

- **LogReg L1** — L1-regularized logistic regression (feature selection via sparsity)
- **RandomForest** — Ensemble of 300 trees with class balancing
- **GradBoost** — sklearn gradient boosting, max_depth=3
- **LightGBM** — Lightweight gradient boosting with class weights
- **XGBoost** — Extreme gradient boosting, max_depth=4

### Validation

- **Stratified 5-Fold CV** if N ≥ 30 and min class ≥ 5
- **Leave-One-Out CV** otherwise (recommended for small samples)

**Metrics reported:**
- ROC-AUC, PR-AUC (thresholds)
- F1, Precision, Recall (at default threshold = 0.5)

### Interpretability

**SHAP analysis:** TreeExplainer on best-performing RandomForest model ranks feature importance by mean |SHAP value|.

Output:
- Beeswarm plot: individual sample SHAP contributions
- Bar plot: average absolute importance
- SHAP value ranking: which features matter most?

See DOCS_TECHNICAL.md § 6 for full model configuration and CV strategy.

## Data Sources

- **Planeta.ru** — primary dataset; parsed from JSON into multi-table structure
- **Boomstarter** — secondary dataset; schema documented in `Readme_diploma.md`

Both datasets cover Russian reward-based crowdfunding campaigns. Success is defined as reaching the funding target.
