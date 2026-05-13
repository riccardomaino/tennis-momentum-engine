import os
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


#####---- Constants
DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR,"cleaned_matches.csv")

SERVE_COLS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"] # prefix: "w_"=winner, "l_"=loser
INVALID_SCORE_TOKENS = {"W/O", "RET", "DEF", "ABD", "UNF", "NAN", ""}


#####---- Helper
def _div(numerator: float, denominator: float, fill: float = np.nan) -> float:
  """
  Divide two numbers and return `fill` instead of raising ZeroDivisionError
  or producing an inf/nan

  Args:
    numerator (float): Number at the numerator of the division
    denominator (float): Number at the denominator of the division
    fill (float, optional): Return value in errors cases (eg. nan values, zero division). Defaults to np.nan.

  Returns:
    float: Result of the division or `fill` if 
  """
  # Suppress NumPy floating-point warning when dividing by zero
  with np.errstate(divide="ignore", invalid="ignore"):
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
      return fill
    return numerator/denominator

def g(prefix: str, col: str) -> str:
  """
  Define the full name of a DataFrame column
  Args:
      prefix (str): Prefix of a certain DataFrame column
      col (str): Specific name of the DataFrame  column 

  Returns:
      str: A string consisting in the full name of the DataFrame column
  """
  return prefix+col


####---- Core Feature Engineering
def build_match_features(df: pd.DataFrame) -> pd.DataFrame:
  """
  Build one feature row per match. The function iterates with iterrows() because the per-row 
  logic involves several conditional branches that are cleaner in Python than in vectorised pandas.

  For each match we compute:
    - Serve rates for winner and loser (ace %, df %, 1st serve %, etc.)
    - Return points won (= 1 - opponent serve points won)
    - Delta features (winner minus loser)
    - Context dummies (surface, tournament level, best-of format)
    - Two binary labels: came_back and upset

  Args
    df (pd.DataFrame): Raw matches from data_retrieval.py

  Returns:
    pd.DataFrame: One row per valid match, all floats or ints, no object cols
  """
  # Binary Cameback
  cleaned_scores = df["score"].str.replace(r"\(\d+\)", "", regex=True)
  first_sets = cleaned_scores.str.split().str[0]
  set_split = first_sets.str.split("-", expand=True)
  w_s1_games = pd.to_numeric(set_split[0], errors="coerce")
  l_s1_games = pd.to_numeric(set_split[1], errors="coerce")
  df["came_back"] = (w_s1_games < l_s1_games).astype(int)
  # Binary Upset
  df["upset"] = np.where(df["winner_rank"] > df["loser_rank"], 1, 0)
  # Winner & Loser Serve %
  df["w_ace_rate"] = df["w_ace"] / df["w_svpt"]
  df["w_df_rate"] = df["w_df"] / df["w_svpt"]
  df["w_1stIn_pct"] = df["w_1stIn"] / df["w_svpt"]
  df["w_1stWon_pct"] = df["w_1stWon"] / df["w_1stIn"]
  df["w_2ndWon_pct"] = df["w_2ndWon"] / (df["w_svpt"] - df["w_1stIn"]) 
  df["w_srvWon_pct"] = (df["w_1stWon"] + df["w_2ndWon"]) / df["w_svpt"]
  df["l_ace_rate"] = df["l_ace"] / df["l_svpt"]
  df["l_df_rate"] = df["l_df"] / df["l_svpt"]
  df["l_1stIn_pct"] = df["l_1stIn"] / df["l_svpt"]
  df["l_1stWon_pct"] = df["l_1stWon"] / df["l_1stIn"]
  df["l_2ndWon_pct"] = df["l_2ndWon"] / (df["l_svpt"] - df["l_1stIn"]) 
  df["l_srvWon_pct"] = (df["l_1stWon"] + df["l_2ndWon"]) / df["l_svpt"]
  # Breakpoints Saved & Conversion %
  df["w_bpSaved_pct"] = df["w_bpSaved"] / df["w_bpFaced"]
  df["l_bpSaved_pct"] = df["l_bpSaved"] / df["l_bpFaced"]
  df["w_bpConversion_pct"] = 1 - df["l_bpSaved_pct"] 
  df["l_bpConversion_pct"] = 1 - df["w_bpSaved_pct"]
  # Return Points Won %
  df["w_retWon_pct"] = 1 - df["l_srvWon_pct"] 
  df["l_retWon_pct"] = 1 - df["w_srvWon_pct"]
  # Ranking Gap
  df["rank_gap"] = df["winner_rank"] - df["loser_rank"]
  # Deltas
  df["delta_ace_pct"] = df["w_ace_rate"] - df["l_ace_rate"]
  df["delta_1stWon_pct"] = df["w_1stWon_pct"] - df["l_1stWon_pct"]
  df["delta_bpSaved_pct"] = df["w_bpSaved_pct"] - df["l_bpSaved_pct"]
  df["delta_srvWon_pct"] = df["w_srvWon_pct"] - df["l_srvWon_pct"]
  df["delta_retWon_pct"] = df["w_retWon_pct"] - df["l_retWon_pct"]
  # Binary One-Hot Context Features
  df = df.assign(
    surf_hard = (df["surface"] == "Hard").astype(int),
    surf_clay = (df["surface"] == "Clay").astype(int),
    surf_grass = (df["surface"] == "Grass").astype(int),
    surf_carpet = (df["surface"] == "Carpet").astype(int),
    is_gs = (df["tourney_level"] == "G").astype(int),
    is_masters = (df["tourney_level"] == "M").astype(int),
    is_atp = (df["tourney_level"] == "A").astype(int),
    is_finals = (df["tourney_level"] == "F").astype(int),
    is_davis = (df["tourney_level"] == "D").astype(int),
    best_of_5 = (df["best_of"] == 5).astype(int)
  )
  
  
  pass



if __name__ == "__main__":
  if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError("Run data_cleaning.py first.")
  
  df = pd.read_csv(INPUT_CSV, low_memory=False)
  print(f"Loaded {len(df):,} cleaned matches\n")
  
  mf = build_match_features(df)
  mf.to_csv("data/match_features.csv", index=False)
  print("→ data/match_features.csv\n")
  
  # pp = build_player_profiles(df)
  # pp.to_csv("data/player_profiles.csv", index=False)
  # print("→ data/player_profiles.csv")