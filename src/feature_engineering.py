import os
import re
import warnings
from typing import Optional, cast

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


#### ----------------- Constants -----------------
DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR,"cleaned_matches.csv")
OUTPUT_MATCHES_CSV = os.path.join(DATA_DIR,"matches_data.csv")
OUTPUT_PLAYERS_CSV = os.path.join(DATA_DIR,"players_data.csv")
SERVE_COLS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "bpSaved", "bpFaced"]
INVALID_SCORE_TOKENS = {"W/O", "RET", "DEF", "ABD", "UNF", "NAN", ""}




#### ------------------ Helper ------------------
def _div(numerator: float, denominator: float, fill: float = np.nan) -> float:
  """Divide two numbers and return `fill` instead of raising ZeroDivisionError producing an inf/nan

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
  """Define the full name of a DataFrame column.
  
  Args:
    prefix (str): Prefix of a certain DataFrame column
    col (str): Specific name of the DataFrame  column 

  Returns:
    str: A string consisting in the full name of the DataFrame column
  """
  return prefix+col

def extract_winner_lost_s1(scores: pd.Series) -> pd.Series:
  """Vectorized function that returns a series with 1s if the winner lost the first set, 0 otherwise.

  Args:
    scores (pd.Series): A series of score
    
  Returns:
    pd.Series: A series of 1s and 0s based on if the winner lost the first set
  """
  first_set = scores.str.split(n=1, expand=True)[0]
  pattern = r"^(\d+)(?:\([0-9]+\))?-(\d+)(?:\([0-9]+\))?"
  extracted = first_set.str.extract(pattern, expand=True)
  w_s1_games = pd.to_numeric(extracted[0], errors="coerce")  # Winner's games in set 1
  l_s1_games = pd.to_numeric(extracted[1], errors="coerce")  # Loser's games in set 1
  winner_lost_s1 = w_s1_games < l_s1_games
  winner_lost_s1 = winner_lost_s1.fillna(False)
  return winner_lost_s1.astype(int)




#### ----------- Features Engineering -----------
def build_match_features(df: pd.DataFrame) -> pd.DataFrame:
  """Build new matches features as columns of the provided DataFrame using Pandas vectorization capabilities.

  For each match we compute:
    - Serve rates for winner and loser (ace %, df %, 1st serve %, etc.)
    - Return points won (= 1 - opponent serve points won)
    - Delta features (winner minus loser)
    - Context dummies (surface, tournament level, best-of format)
    - Two binary labels: came_back and upset

  Args
    df (pd.DataFrame): The input DataFrame containing matches data

  Returns:
    pd.DataFrame: The output DataFrame with the new features
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
  df["rank_gap"] = df["loser_rank"] - df["winner_rank"]
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
  
  df = df.replace([np.inf, -np.inf], np.nan)
  df = df.dropna(subset=["w_ace_rate", "l_ace_rate", "came_back"])
  # df = df.drop(columns=["surface", "draw_size", "tourney_level", "match_num", "best_of", "round",
  #                       "winner_seed", "winner_entry", "winner_ioc", "winner_rank_points",
  #                       "loser_seed", "loser_entry", "loser_ioc", "loser_rank_points"])
  print(f"[matches_data] {len(df):,} rows  |  "
        f"comeback rate: {df["came_back"].mean():.2%} "
        f"upset rate: {df["upset"].mean():.2%}")
  return df

def build_player_profiles(df: pd.DataFrame, min_matches: int = 100) -> pd.DataFrame:
  """Builds a new DataFrame containing per-player career features

  Args:
    df (pd.DataFrame): The input DataFrame containing matches data
    min_matches (int, optional): Minimum number of matches to consider a player. Defaults to 30.
    
  Returns:
    pd.DataFrame: The output DataFrame with the per-player career features 
  """
  winner_lost_s1 = extract_winner_lost_s1(df["score"])
  
  roles = [
    ("winner", "w_", "l_", 1),
    ("loser", "l_", "w_", 0)
  ]
  results = []
  for role, pfx, opfx, won in roles:
    player_df = pd.DataFrame({
      "player": df[f"{role}_name"],
      "won": won,
      "year": pd.to_numeric(df["year"], errors="coerce"),
      "surface": df["surface"],
      "rank": pd.to_numeric(df[f"{role}_rank"], errors="coerce"),
      # Serve Stats
      "svpt": pd.to_numeric(df[f"{pfx}svpt"], errors="coerce"),
      "ace": pd.to_numeric(df[f"{pfx}ace"], errors="coerce"),
      "df": pd.to_numeric(df[f"{pfx}df"], errors="coerce"),
      "1stIn": pd.to_numeric(df[f"{pfx}1stIn"], errors="coerce"),
      "1stWon": pd.to_numeric(df[f"{pfx}1stWon"], errors="coerce"),
      "2ndWon": pd.to_numeric(df[f"{pfx}2ndWon"], errors="coerce"),
      "bpSaved": pd.to_numeric(df[f"{pfx}bpSaved"], errors="coerce"),
      "bpFaced": pd.to_numeric(df[f"{pfx}bpFaced"], errors="coerce"),
      # Opponent Stats
      "o_svpt": pd.to_numeric(df[f"{opfx}svpt"], errors="coerce"),
      "o_1stWon": pd.to_numeric(df[f"{opfx}1stWon"], errors="coerce"),
      "o_2ndWon": pd.to_numeric(df[f"{opfx}2ndWon"], errors="coerce"),
      "o_bpFaced": pd.to_numeric(df[f"{opfx}bpFaced"], errors="coerce"),
      "o_bpSaved": pd.to_numeric(df[f"{opfx}bpSaved"], errors="coerce")
    })
    # Comeback opportunity & success
    player_df["cb_opp"] = (((won == 1) & winner_lost_s1) | 
                           ((won == 0) & ~winner_lost_s1)).astype(int)
    player_df["cb_success"] = ((won == 1) & winner_lost_s1).astype(int)
    results.append(player_df)
  
  # Combine winner/loser rows, remove rows without valid player name
  players_stats_df = pd.concat(results, axis=0, ignore_index=True)
  players_stats_df = cast(pd.DataFrame, players_stats_df)
  if players_stats_df.empty:
    return pd.DataFrame()
  
  grp = players_stats_df.groupby("player")

  # Aggregate raw counts
  agg = grp.agg(
    matches_played = ("won", "count"),
    wins = ("won", "sum"),
    cb_opportunities = ("cb_opp", "sum"),
    cb_successes = ("cb_success", "sum"),
    sum_svpt = ("svpt", "sum"),
    sum_ace  = ("ace", "sum"),
    sum_df = ("df", "sum"),
    sum_1stIn = ("1stIn", "sum"),
    sum_1stWon = ("1stWon", "sum"),
    sum_2ndWon = ("2ndWon", "sum"),
    sum_bpSaved = ("bpSaved", "sum"),
    sum_bpFaced = ("bpFaced",  "sum"),
    sum_o_svpt= ("o_svpt","sum"),
    sum_o_1stWon  = ("o_1stWon", "sum"),
    sum_o_2ndWon  = ("o_2ndWon",  "sum"), 
    sum_o_bpFaced = ("o_bpFaced","sum"),
    sum_o_bpSaved = ("o_bpSaved", "sum"),
    avg_rank = ("rank", "mean"),
    best_rank = ("rank", "min")
  ).reset_index()

  # Filter minimum matches
  agg = agg[agg["matches_played"] >= min_matches].copy().reset_index(drop=True)
  
  # Compute rates from aggregated counts
  agg["win_rate"] = agg["wins"] / agg["matches_played"]
  agg["comeback_rate"] = np.where(
    agg["cb_opportunities"] >= 10, 
    agg["cb_successes"] / agg["cb_opportunities"],
    np.nan
  )
  agg["ace_rate"] = agg["sum_ace"] / agg["sum_svpt"]
  agg["df_rate"] = agg["sum_df"] / agg["sum_svpt"]
  agg["first_serve_in"] = agg["sum_1stIn"] / agg["sum_svpt"]
  agg["first_serve_won"] = agg["sum_1stWon"] / agg["sum_1stIn"]
  agg["second_serve_won"] = agg["sum_2ndWon"] / (agg["sum_svpt"] - agg["sum_1stIn"])
  agg["srv_pts_won"] = (agg["sum_1stWon"] + agg["sum_2ndWon"]) / agg["sum_svpt"]
  agg["bp_save_rate"] = agg["sum_bpSaved"] / agg["sum_bpFaced"]
  agg["ret_pts_won"] = 1 - (agg["sum_o_1stWon"] + agg["sum_o_2ndWon"]) / agg["sum_o_svpt"]
  agg["bp_conv_rate"] = (agg["sum_o_bpFaced"] - agg["sum_o_bpSaved"]) / agg["sum_o_bpFaced"]

  # Clutch Index: z-score average of bp_save_rate & bp_conv_rate
  for col in ["bp_save_rate", "bp_conv_rate"]:
    mu = agg[col].mean()
    sd = agg[col].std()
    agg[f"_{col}_z"] = (agg[col] - mu) / sd
  agg["clutch_index"] = (agg["_bp_save_rate_z"] + agg["_bp_conv_rate_z"]) / 2

  # Serve style score: 0="grinder", 1="big server"
  for col in ["ace_rate", "first_serve_in"]:
    mn = agg[col].min()
    mx = agg[col].max()
    agg[f"_{col}_n"] = (agg[col] - mn) / (mx - mn + 1e-9)
  agg["serve_style_score"] = (agg["_ace_rate_n"] + agg["_first_serve_in_n"]) / 2

  # Clean up temp columns
  drop_cols = [c for c in agg.columns if c.startswith(("sum_", "_"))]
  agg.drop(columns=drop_cols, inplace=True)

  print(f"[players_data] {len(agg):,} rows  |  (≥{min_matches} matches)")
  return agg
  


#### ------------------- Main -------------------
if __name__ == "__main__":
  if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError("Run data_cleaning.py first.")
  
  df = pd.read_csv(INPUT_CSV, low_memory=False)
  print(f"Loaded {len(df):,} cleaned matches\n")
  
  mf = build_match_features(df)
  mf.to_csv(OUTPUT_MATCHES_CSV, index=False)
  print(f"Saved [matches_data] dataset → {OUTPUT_MATCHES_CSV}\n")
  
  pp = build_player_profiles(df)
  pp.to_csv(OUTPUT_PLAYERS_CSV, index=False)
  print(f"Saved [players_data] dataset → {OUTPUT_PLAYERS_CSV}")