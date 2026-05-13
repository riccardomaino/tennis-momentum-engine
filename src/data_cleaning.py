import pandas as pd
import numpy as np
import re
import os

#####---- Constants
OUTPUT_DIR ="data"
INPUT_CSV = os.path.join(OUTPUT_DIR, "raw_matches.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "cleaned_matches.csv")
INVALID_SCORE_TOKENS = {"W/O", "RET", "DEF", "ABD", "UNF", "NAN", ""}
NUMERIC_COLS = ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced", "w_SvGms",
            "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced", "l_SvGms",
            "winner_rank", "winner_rank_points", "loser_rank_points", "loser_rank",
            "winner_age", "winner_ht", "loser_age", "loser_ht", 
            "best_of"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
  """
  Cleans a tennis match dataset by applying standard filtering and data type conversions.

  Args:
    df (pd.DataFrame): Raw tennis match DataFrame

  Returns:
    pd.DataFrame: Cleaned DataFrame with filtered rows and properly typed columns.
  """
  
  # Filter out matches with no score
  df = df[df["score"].notna()]
  # Filter out retirements, walkovers, defaults (stats are skewed/incomplete)
  df = df[~df["score"].str.contains(INVALID_SCORE_TOKENS, case=False, na=False)].copy()
  # Handle missing `surface` value
  df["surface"] = df["surface"].fillna("Unknown")
  # Convert all the numeric columns to numeric
  for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
  # Convert dates 
  df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
  return df

if __name__ == "__main__":
  df = pd.read_csv(INPUT_CSV)
  df_cleaned = clean_data(df)
  df_cleaned.to_csv(OUTPUT_CSV, index=False)
  print(f"\nSaved {len(df_cleaned):,} matches → {OUTPUT_CSV}")
  print(f"    Columns : {list(df_cleaned.columns[:10])} ... ({df_cleaned.shape[1]} total)")
  print(f"    Years   : {df_cleaned["year"].min()} - {df_cleaned["year"].max()}")
  print(f"    Memory  : {df_cleaned.memory_usage(deep=True).sum() / 1e6:.1f} MB")
  