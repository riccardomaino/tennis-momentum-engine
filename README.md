# 🎾 Tennis Momentum Engine

> **A data-driven profiling and prediction system for ATP Tour comeback, clutch performance, and upset dynamics — built on 20+ years of match data.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Table of Contents

- [🎾 Tennis Momentum Engine](#-tennis-momentum-engine)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Key Features](#key-features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install uv](#install-uv)
    - [Clone and install](#clone-and-install)
    - [Dependencies](#dependencies)
  - [Running the Pipeline](#running-the-pipeline)
  - [Feature Dictionary](#feature-dictionary)
    - [Match Features (`matches_data.csv`)](#match-features-matches_datacsv)
      - [Identifiers](#identifiers)
      - [Labels](#labels)
      - [Winner serve features (`w_*`)](#winner-serve-features-w_)
      - [Loser serve features (`l_*`)](#loser-serve-features-l_)
      - [Delta features](#delta-features)
      - [Context dummies](#context-dummies)
    - [Player Profiles (`players_data.csv`)](#player-profiles-players_datacsv)
  - [Limitations](#limitations)
  - [Data Source](#data-source)
  - [📄 License](#-license)

---
## Project Overview

The **Tennis Momentum Engine** answers three related questions using 30+ years of ATP Tour data:

| Question | Task | Output |
|---|---|---|
| What makes a player a "comeback player"? | Binary classification | Comeback probability |
| Can we predict upsets before they happen? | Binary classification | Upset probability + feature ranking |
| Which players perform best under pressure? | Aggregation + clustering | Clutch Index leaderboard + player radar |

The project was built as a full end-to-end data science / machine learning pipeline — from raw CSV ingestion through to fully trained models.

---

## Key Features
- **Two ML classifiers** (Comeback + Upset) with cross-validated model selection and ROC/Confusion Matrix evaluation
- **Player profile engine** computing Clutch Index (z-scored break-point performance), Comeback Rate, and Serve Style Score for every ATP player with >= 100 matches
- **Interactive Streamlit app** with 5 tabs: EDA overview, comeback predictor, upset predictor, player lookup (with radar chart), and sortable leaderboards
- **Six EDA Plots** useful insight about the data

---

## Installation

### Prerequisites

- Python 3.10+
- uv

### Install uv

If you don't already have `uv` installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/tennis_momentum_engine.git
cd tennis_momentum_engine

uv venv
source .venv/bin/activate # Windows: .venv\Scripts\activate

uv sync
```

### Dependencies

Dependencies are managed through `pyproject.toml`.

Example:

```toml
[project]
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.4",
    "xgboost>=2.0",
    "shap>=0.44",
    "streamlit>=1.32",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "plotly>=5.18",
    "requests>=2.31",
    "tqdm>=4.66",
    "joblib>=1.3"
]
```

---

## Running the Pipeline

Run each step in order. Each script is standalone and can be re-run independently.

```bash
# Step 1 — Download ATP matches
uv run src.data_retrieval.py

# Step 2 — Build feature tables
uv run src/feature_engineering.py

# Step 3 — Generate EDA plots
uv run src/eda.py

# Step 4 — Train and evaluate models
uv run src/train.py
```

---

## Feature Dictionary

### Match Features (`matches_data.csv`)

#### Identifiers
| Column | Description |
|---|---|
| `winner_name` / `loser_name` | Player names |
| `surface` | Clay / Hard / Grass / Carpet |
| `tourney_level` | G=Grand Slam, M=Masters, A=250/500 |
| `round` | R128, R64, QF, SF, F … |
| `year` | Calendar year |

#### Labels
| Column | Description |
|---|---|
| `came_back` | 1 if the winner lost Set 1 (comeback win), 0 otherwise |
| `upset` | 1 if the lower-ranked player won, 0 otherwise |

#### Winner serve features (`w_*`)
| Column | Formula |
|---|---|
| `w_ace_rate` | aces / serve points |
| `w_df_rate` | double faults / serve points |
| `w_1stIn_pct` | 1st serves in / serve points |
| `w_1stWon_pct` | 1st serve points won / 1st serves in |
| `w_2ndWon_pct` | 2nd serve points won / 2nd serve points |
| `w_bpSaved_pct` | break points saved / break points faced |
| `w_srvWon_pct` | (1st won + 2nd won) / serve points |
| `w_retWon_pct` | 1 − opponent serve points won rate |

#### Loser serve features (`l_*`)
Same schema as winner features with `l_` prefix.

#### Delta features
| Column | Description |
|---|---|
| `delta_ace_pct` | `w_ace_rate − l_ace_rate` |
| `delta_1stWon_pct` | `w_1stWon_pct − l_1stWon_pct` |
| `delta_bpSaved_pct` | `w_bpSaved_pct − l_bpSaved_pct` |
| `delta_srvWon_pct` | `w_srvWon_pct − l_srvWon_pct` |
| `delta_retWon_pct` | `w_retWon_pct − l_retWon_pct` |
| `rank_gap` | `loser_rank − winner_rank` (positive = winner ranked higher) |

#### Context dummies
| Column | Value |
|---|---|
| `surf_hard` / `surf_clay` / `surf_grass` / `surf_carpet` | 1 if surface matches, 0 otherwise (Other = all zeros) |
| `is_gs` | 1 if Grand Slam |
| `is_masters` | 1 if Masters 1000 |
| `is_atp` | 1 if ATP 250/500 |
| `is_finals` | 1 if ATP Finals |
| `is_davis` | 1 if Davis Cup |
| `best_of_5` | 1 if best-of-5 format |

### Player Profiles (`players_data.csv`)

| Column | Description |
|---|---|
| `player` | Player name |
| `matches_played` | Total career matches in dataset |
| `win_rate` | Career win rate |
| `cb_opportunities` | Matches where player either lost Set 1 (as winner) or lost the match |
| `cb_successes` | Matches where player won after losing Set 1 |
| `comeback_rate` | `cb_successes / cb_opportunities` (NaN if < 10 opportunities) |
| `clutch_index` | z-score average of `bp_save_rate` and `bp_conv_rate` |
| `serve_style_score` | Normalised 0–1 composite of `ace_rate` + `first_serve_in` |
| `bp_conv_rate` | Break points converted / break points created |
| `avg_rank` / `best_rank` | Career average and best ATP ranking |

---

## Limitations
1. **No set-level stats** — all features are match aggregates. The model profiles comeback *types of players*, not comeback *situations*.
2. **Era effects** — serving styles and ranking systems changed significantly between 1991 and 2023.

---

## Data Source

All match data comes from **Jeff Sackmann's open-source ATP dataset**:

| Column | Description |
|---|---|
| `tourney_id` | Unique tournament identifier (`year-tournament_code`). |
| `tourney_name` | Name of the tournament. |
| `surface` | Court surface type (`Hard`, `Clay`, `Grass`, `Carpet`). |
| `draw_size` | Number of players in the tournament draw. |
| `tourney_level` | Tournament category/level (`G`=Grand Slam, `M`=Masters, `A`=ATP, `D`=Davis Cup.). |
| `tourney_date` | Tournament start date in `YYYYMMDD` format. |
| `match_num` | Match number within the tournament. |
| `winner_id` | Unique player ID of the winner. (eg. `104676`)|
| `winner_seed` | Tournament seed of the winner. (eg. `5.0` or `NaN`) |
| `winner_entry` | Entry type of the winner (`Q`=Qualifier, `WC`=Wildcard, `LL`=Lucky Loser, `SE`=Special Exempt). |
| `winner_name` | Full name of the winner. (eg. `Casper Ruud`)|
| `winner_hand` | Dominant hand of the winner (`R`=Right, `L`=Left, `U`=Unknown). |
| `winner_ht` | Height of the winner in centimeters. (eg. `188.0`)|
| `winner_ioc` | Country code (IOC format) of the winner. (eg. `FRA` or `ITA`)|
| `winner_age` | Age of the winner at match time. (eg. `27.[0-9]`) |
| `loser_id` | Unique player ID of the loser. |
| `loser_seed` | Tournament seed of the loser. |
| `loser_entry` | Entry type of the loser. |
| `loser_name` | Full name of the loser. |
| `loser_hand` | Dominant hand of the loser. |
| `loser_ht` | Height of the loser in centimeters. |
| `loser_ioc` | Country code (IOC format) of the loser. |
| `loser_age` | Age of the loser at match time. |
| `score` | Final match score. (eg. `6-4 7-6(2)` or `6-4 6-2 RET`)|
| `best_of` | Number of sets required to win the match (`3` or `5`). |
| `round` | Tournament round (`RR`, `R128`, `R64`, `R32`, `QF`, `SF`, `F`). |
| `minutes` | Match duration in minutes. (eg. `124.0` or `61.0`) |
| `w_ace` | Number of aces served by the winner. |
| `w_df` | Number of double faults committed by the winner. |
| `w_svpt` | Total service points played by the winner. |
| `w_1stIn` | Number of first serves made by the winner. |
| `w_1stWon` | Number of points won on first serve by the winner. |
| `w_2ndWon` | Number of points won on second serve by the winner. |
| `w_SvGms` | Number of service games played by the winner. |
| `w_bpSaved` | Number of break points saved by the winner. |
| `w_bpFaced` | Number of break points faced by the winner. |
| `l_ace` | Number of aces served by the loser. |
| `l_df` | Number of double faults committed by the loser. |
| `l_svpt` | Total service points played by the loser. |
| `l_1stIn` | Number of first serves made by the loser. |
| `l_1stWon` | Number of points won on first serve by the loser. |
| `l_2ndWon` | Number of points won on second serve by the loser. |
| `l_SvGms` | Number of service games played by the loser. |
| `l_bpSaved` | Number of break points saved by the loser. |
| `l_bpFaced` | Number of break points faced by the loser. |
| `winner_rank` | ATP ranking of the winner at match time. |
| `winner_rank_points` | ATP ranking points of the winner. |
| `loser_rank` | ATP ranking of the loser at match time. |
| `loser_rank_points` | ATP ranking points of the loser. |
| `year` | Year of the tournament/match. |

> Sackmann, J. (2024). *tennis_atp*. GitHub repository.
> https://github.com/JeffSackmann/tennis_atp

The dataset is provided under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

> Please respect the license terms: attribution is required, commercial use is not permitted.

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

The underlying data (Jeff Sackmann's ATP dataset) is subject to its own CC BY-NC-SA 4.0 license.

---







