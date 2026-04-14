# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Master's thesis research project analyzing success factors of reward-based crowdfunding campaigns on Russian platforms (Planeta.ru and Boomstarter). The goal is to identify which campaign characteristics (text, media, financial, temporal, author) statistically predict campaign success.

## Running the Analysis

This is a Jupyter notebook project with no build system. Run analysis in JupyterLab or VS Code:

```bash
jupyter lab Planeta.ipynb
# or
jupyter notebook Planeta.ipynb
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

## Data Sources

- **Planeta.ru** — primary dataset; parsed from JSON into multi-table structure
- **Boomstarter** — secondary dataset; schema documented in `Readme_diploma.md`

Both datasets cover Russian reward-based crowdfunding campaigns. Success is defined as reaching the funding target.
