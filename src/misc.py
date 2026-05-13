import os
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


#####---- Constants
DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR,"raw_matches.csv")
OUTPUT_MATCHES_CSV = os.path.join(DATA_DIR,"matches_features.csv")
OUTPUT_PLAYER_CSV = os.path.join(DATA_DIR,"player_profile.csv")

SERVE_COLS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"] # prefix: "w_"=winner, "l_"=loser
INVALID_SCORE_TOKENS = {"W/O", "RET", "DEF", "ABD", "UNF", "NAN", ""}

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

def parse_score(score: str) -> Optional[dict]:
  """
  Parse a score string such as '3-6 7-5 6-4' or '7-6(4) 1-6 6-3 6-4' to obtain match score infos.

  Args:
    score (str): A text that describe the match score

  Returns:
    Optional[dict]: A dict with match score infos, None if the score cannot be parsed reliably
  """
  # Check if the score can be reliably parsed and manipulate it (eg. remove tie-breaks infos)
  if not isinstance(score, str):
    return None
  
  tokens = score.strip().upper()
  if any(invalid_t in tokens for invalid_t in INVALID_SCORE_TOKENS):
    return None

  s_cleaned = re.sub(r"\(\d+\)", score, "").strip()
  s_parts = s_cleaned.split()
  
  if len(s_parts) < 2:
    return None
  
  # Extract infos from the score string
  w_sets = l_sets = 0
  winner_lost_s1 = False
  
  for i, part in enumerate(s_parts):
    try:
      w_games, l_games = map(int, part.split("-"))
    except (ValueError, AttributeError):
      return None
    
    if w_games > l_games:
      w_sets += 1
    else:
      l_sets += 1
      if i == 0:
        winner_lost_s1 = True

  total_sets = w_sets + l_sets
  if total_sets < 2:
    return None #  incomplete/corrupted entry
  
  return {
    "winner_sets": w_sets,
    "loser_sets": l_sets,
    "total_sets": total_sets,
    "winner_lost_s1": winner_lost_s1
  }

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
  records = []
  for _, row in df.iterrows():
    # Parse score
    score_parsed = parse_score(row.get("score"))
    if score_parsed is None:
      continue
    
    # Helper function to fetch numeric cell from a row (NaN for invalid parsing)
    def g(prefix: str, col: str) -> float:
      return pd.to_numeric(row.get(f"{prefix}{col}"), errors="coerce")

    # Feature: Winner Serve Stats %
    w_svpt = g("w_", "svpt")
    w_ace = g("w_", "ace")
    w_df = g("w_", "df")
    w_1stIn = g("w_", "1stIn")
    w_1stWon = g("w_", "1stWon")
    w_2ndWon = g("w_", "2ndWon")
    w_bpSaved = g("w_", "bpSaved")
    w_bpFaced = g("w_", "bpFaced")
    w_ace_rate = _div(w_ace, w_svpt)
    w_df_rate = _div(w_df, w_svpt)
    w_1stIn_pct = _div(w_1stIn, w_svpt)
    w_1stWon_pct = _div(w_1stWon, w_1stIn)
    w_2ndWon_pct = _div(w_2ndWon, w_svpt - w_1stIn)
    w_srvWon_pct = _div(w_1stWon + w_2ndWon, w_svpt)
    
    # Feature: Loser Serve Stats %
    l_svpt = g("l_", "svpt")
    l_ace  = g("l_", "ace")
    l_df = g("l_", "df")
    l_1stIn = g("l_", "1stIn")
    l_1stWon = g("l_", "1stWon")
    l_2ndWon = g("l_", "2ndWon")
    l_bpSaved = g("l_", "bpSaved")
    l_bpFaced = g("l_", "bpFaced")
    l_ace_rate = _div(l_ace, l_svpt)
    l_df_rate = _div(l_df, l_svpt)
    l_1stIn_pct = _div(l_1stIn, l_svpt)
    l_1stWon_pct = _div(l_1stWon, l_1stIn)
    l_2ndWon_pct = _div(l_2ndWon, l_svpt - l_1stIn)
    l_srvWon_pct = _div(l_1stWon + l_2ndWon, l_svpt)
    
    # Feature: Beakpoints Stats %
    w_bpSaved_pct = _div(w_bpSaved, w_bpFaced)
    l_bpSaved_pct = _div(l_bpSaved, l_bpFaced)
    w_bpConversion_pct = 1 - l_bpSaved_pct if not pd.isna(l_bpSaved_pct) else np.nan
    l_bpConversion_pct = 1 - w_bpSaved_pct if not pd.isna(w_bpSaved_pct) else np.nan
    
    # Features: Return Points Won %
    w_retWon_pct = 1 - l_srvWon_pct if not pd.isna(l_srvWon_pct) else np.nan
    l_retWon_pct = 1 - w_srvWon_pct if not pd.isna(w_srvWon_pct) else np.nan

    # Features: Ranking Gap
    w_rank = g("winner", "rank")
    l_rank = g("loser", "rank")
    rank_gap = w_rank - l_rank if not (pd.isna(w_rank) or pd.isna(l_rank)) else np.nan
    
    # Features: Binary Features (Comeback & Upset)
    came_back = int(score_parsed["winner_lost_s1"])
    upset = int(w_rank > l_rank) if not pd.isna(rank_gap) else np.nan
    
    # Features: Binary Features (Surface Context, eg. surface -> Unknown=[0 0 0 0], hard=[1 0 0 0] etc.)
    surface = str(row.get("surface", "")).strip()
    surf_clay = int(surface == "Clay")
    surf_grass = int(surface == "Grass")
    surf_carpet = int(surface == "Carpet")
    
    # Features: Binary Features (Tournament Context, eg. tourney_level O=[0 0 0 0 0], G=[1 0 0 0 0], etc.)
    tourney_level = str(row.get("tourney_level", "")).strip()
    is_gs = int(tourney_level == "G")
    is_masters = int(tourney_level == "M")
    is_atp = int(tourney_level == "A")
    is_finals = int(tourney_level == "F")
    is_davis = int(tourney_level == "D")
    
    # Features: Binary Features (Format, eg. best_of_5 = 1)
    best_of_5 = int(pd.to_numeric(row.get("best_of"), errors="coerce") == 5)
    
    # Create a record along with all the new features
    records.append({
      #-- Identifiers
      "tourney_id": row.get("tourney_id"),
      "tourney_name": row.get("tourney_name"),
      "year": pd.to_numeric(row.get("year"), errors="coerce"),
      "winner_name": row.get("winner_name"),
      "loser_name": row.get("loser_name"),
      "winner_rank": w_rank,
      "loser_rank": l_rank,
      "surface": surface,
      "tourney_level": tourney_level,
      "round": row.get("round"),
      "score": row.get("score"),
      "total_sets": score_parsed["total_sets"],
      "best_of_5": best_of_5,
      #-- Comeback & Upset Features
      "came_back": came_back,
      "upset": upset,
      #-- Winner Features
      "w_ace_rate": w_ace_rate, "w_df_rate": w_df_rate,
      "w_1stIn_pct": w_1stIn_pct, "w_1stWon": w_1stWon_pct,
      "w_2ndWon_pct": w_2ndWon_pct, "w_bpSaved_pct": w_bpSaved_pct,
      "w_srvWon_pct": w_srvWon_pct, "w_retWon_pct": w_retWon_pct,
      "w_bpConversion_pct": w_bpConversion_pct,
      #-- Loser Features
      "l_ace_rate": l_ace_rate, "l_df_rate": l_df_rate,
      "l_1stIn_pct": l_1stIn_pct, "l_1stWon": l_1stWon_pct,
      "l_2ndWon_pct": l_2ndWon_pct, "l_bpSaved_pct": l_bpSaved_pct,
      "l_srvWon_pct": l_srvWon_pct, "l_retWon_pct": l_retWon_pct,
      "l_bpConversion_pct": l_bpConversion_pct,
      #-- Delta Features
      "delta_ace_pct": w_ace_rate - l_ace_rate,
      "delta_1stWon_pct": w_1stWon_pct - l_1stWon_pct,
      "delta_bpSaved_pct": w_bpSaved_pct - l_bpSaved_pct,
      "delta_srvWon_pct": w_srvWon_pct - l_srvWon_pct,
      "delta_retWon_pct": w_retWon_pct - l_retWon_pct,
      "rank_gap": rank_gap,
      #-- Context Features
      "surf_clay": surf_clay, "surf_grass": surf_grass, "surf_carpet": surf_carpet,
      "is_gs": is_gs, "is_masters": is_masters, "is_atp": is_atp,
      "is_finals": is_finals, "is_davis": is_davis
    })
  
  out = pd.DataFrame(records)
  print(f"[match_features] {len(out):,} rows  |  "
        f"comeback rate: {out["came_back"].mean():.2%}"
        f"upset rate: {out["upset"].mean():.2%}")
  return out

def build_player_profiles(df: pd.DataFrame,
                          min_matches: int = 40) -> pd.DataFrame:
    """
    Aggregate career statistics for every player in the dataset.

    Approach
    --------
    We "unstack" each match row into TWO player rows — one for the winner,
    one for the loser — then groupby player name and sum the raw counts
    before computing rates.  Summing counts then dividing is statistically
    correct; averaging rates would give equal weight to short and long
    matches (Simpson's paradox risk).

    Derived metrics
    ---------------
    comeback_rate   : comebacks / comeback_opportunities  (min 10 opps)
    clutch_index    : z-score average of bp_save_rate and bp_conv_rate.
                      Measures performance under pressure relative to
                      the rest of the tour.
    serve_style_score: normalised composite of ace_rate + first_serve_in.
                      High score → big server; low score → grinder.

    Parameters
    ----------
    df          : pd.DataFrame  Raw matches (not the match_features table)
    min_matches : int           Players with fewer matches are dropped

    Returns
    -------
    pd.DataFrame  One row per qualifying player
    """

    rows = []

    for _, row in df.iterrows():
        parsed = parse_score(row.get("score"))
        if parsed is None:
            continue

        def g(prefix: str, col: str) -> float:
            return pd.to_numeric(row.get(f"{prefix}{col}"), errors="coerce")

        for role, pfx, opfx, won in [
            ("winner", "w_", "l_", 1),
            ("loser",  "l_", "w_", 0),
        ]:
            name = row.get(f"{role}_name")
            if not isinstance(name, str) or not name.strip():
                continue

            svpt  = g(pfx, "svpt");  ace  = g(pfx, "ace");   df_  = g(pfx, "df")
            first = g(pfx, "1stIn"); fw   = g(pfx, "1stWon"); sw  = g(pfx, "2ndWon")
            bps   = g(pfx, "bpSaved"); bpf = g(pfx, "bpFaced")

            # opponent serve (for return / break-point conversion stats)
            o_svpt = g(opfx, "svpt"); o_fw = g(opfx, "1stWon"); o_sw = g(opfx, "2ndWon")
            o_bpf  = g(opfx, "bpFaced");  o_bps = g(opfx, "bpSaved")

            # How many comeback opportunities?
            # Winner: +1 if they were in a comeback (lost s1)
            # Loser:  +1 always (they lost the match, so if they lost s1 too
            #         that's just a normal loss; we count their match losses)
            cb_opp     = int(won == 1 and parsed["winner_lost_s1"]) + int(won == 0)
            cb_success = int(won == 1 and parsed["winner_lost_s1"])

            rows.append({
                "player":      name,
                "won":         won,
                "cb_opp":      cb_opp,
                "cb_success":  cb_success,
                # serve raw counts
                "svpt": svpt, "ace": ace, "df": df_,
                "first": first, "fw": fw, "sw": sw,
                "bps": bps, "bpf": bpf,
                # return / bp conversion raw counts
                "o_svpt": o_svpt, "o_fw": o_fw, "o_sw": o_sw,
                "o_bpf":  o_bpf,  "o_bps": o_bps,
                "rank": pd.to_numeric(row.get(f"{role}_rank"), errors="coerce"),
                "year": pd.to_numeric(row.get("year"), errors="coerce"),
                "surface": row.get("surface"),
            })

    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame()

    grp = long.groupby("player")

    # ── Aggregate raw counts ──────────────────────────────────────────────────
    agg = grp.agg(
      matches_played = ("won", "count"),
      wins           = ("won", "sum"),
      cb_opportunities = ("cb_opp", "sum"),
      cb_successes     = ("cb_success", "sum"),
      sum_svpt  = ("svpt",  "sum"), sum_ace  = ("ace",  "sum"),
      sum_df    = ("df",    "sum"), sum_first = ("first","sum"),
      sum_fw    = ("fw",    "sum"), sum_sw    = ("sw",   "sum"),
      sum_bps   = ("bps",   "sum"), sum_bpf   = ("bpf",  "sum"),
      sum_o_svpt= ("o_svpt","sum"), sum_o_fw  = ("o_fw", "sum"),
      sum_o_sw  = ("o_sw",  "sum"), sum_o_bpf = ("o_bpf","sum"),
      sum_o_bps = ("o_bps", "sum"),
      avg_rank  = ("rank",  "mean"),
      best_rank = ("rank",  "min"),
    ).reset_index()

    # ── Compute rates from aggregated counts ──────────────────────────────────
    agg["win_rate"]          = agg["wins"]      / agg["matches_played"]
    agg["comeback_rate"]     = np.where(
        agg["cb_opportunities"] >= 10,
        agg["cb_successes"] / agg["cb_opportunities"],
        np.nan
    )
    agg["ace_rate"]          = agg["sum_ace"]  / agg["sum_svpt"]
    agg["df_rate"]           = agg["sum_df"]   / agg["sum_svpt"]
    agg["first_serve_in"]    = agg["sum_first"]/ agg["sum_svpt"]
    agg["first_serve_won"]   = agg["sum_fw"]   / agg["sum_first"]
    agg["second_serve_won"]  = agg["sum_sw"]   / (agg["sum_svpt"] - agg["sum_first"])
    agg["srv_pts_won"]       = (agg["sum_fw"]  + agg["sum_sw"]) / agg["sum_svpt"]
    agg["bp_save_rate"]      = agg["sum_bps"]  / agg["sum_bpf"]
    agg["ret_pts_won"]       = 1 - (agg["sum_o_fw"] + agg["sum_o_sw"]) / agg["sum_o_svpt"]
    agg["bp_conv_rate"]      = (agg["sum_o_bpf"] - agg["sum_o_bps"]) / agg["sum_o_bpf"]

    # ── Clutch Index: z-score average of bp_save_rate & bp_conv_rate ─────────
    for col in ["bp_save_rate", "bp_conv_rate"]:
        mu = agg[col].mean(); sd = agg[col].std()
        agg[f"_{col}_z"] = (agg[col] - mu) / sd

    agg["clutch_index"] = (agg["_bp_save_rate_z"] + agg["_bp_conv_rate_z"]) / 2

    # ── Serve style score: 0 = grinder, 1 = big server ───────────────────────
    for col in ["ace_rate", "first_serve_in"]:
        mn = agg[col].min(); mx = agg[col].max()
        agg[f"_{col}_n"] = (agg[col] - mn) / (mx - mn + 1e-9)

    agg["serve_style_score"] = (agg["_ace_rate_n"] + agg["_first_serve_in_n"]) / 2

    # ── Clean up temp columns ─────────────────────────────────────────────────
    drop_cols = [c for c in agg.columns if c.startswith(("sum_", "_"))]
    agg.drop(columns=drop_cols, inplace=True)

    # ── Filter minimum matches ────────────────────────────────────────────────
    agg = agg[agg["matches_played"] >= min_matches].copy().reset_index(drop=True)

    print(f"[player_profiles] {len(agg):,} players  (≥{min_matches} matches)")
    return agg
  
if __name__ == "__main__":
  RAW = "data/raw_matches.csv"
  if not os.path.exists(RAW):
    raise FileNotFoundError("Run data_retrieval.py first.")

  df = pd.read_csv(RAW, low_memory=False)
  print(f"Loaded {len(df):,} raw matches\n")

  mf = build_match_features(df)
  mf.to_csv(OUTPUT_MATCHES_CSV, index=False)
  print("→ data/match_features.csv\n")

  pp = build_player_profiles(df)
  pp.to_csv(OUTPUT_PLAYER_CSV, index=False)
  print("→ data/player_profiles.csv")