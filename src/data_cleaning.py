import pandas as pd
import numpy as np
import re

DATASET_PATH = "data/raw_matches.csv"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
  # Fill NaN of some columns
  df[['winner_entry', 'loser_entry']] = df[['winner_entry', 'loser_entry']].fillna('OK')
  df['surface'] = df['surface'].fillna('Unknown')
  # Filter out no score matches
  df = df[df['score'].notna()] 
  # Filter out NaN values of important stats
  df = df.dropna(subset=['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced'])
  df = df.dropna(subset=['l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'])
  # Filter out retirements, walkovers, defaults (stats are skewed/incomplete)
  df = df[~df["score"].str.contains('RET|W/O|DEF', case=False, na=False)]
  # Convert tourney_date to datetime
  df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
  return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
  records = []

if __name__ == "__main__":
  df = pd.read_csv(DATASET_PATH)
  
  