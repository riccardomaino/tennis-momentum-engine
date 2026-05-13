
import pandas as pd
import os

# Configuration constants
RAW_DATA_PATH = "data/raw_matches.csv"


if __name__ == "__main__":  
  if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Run data_retrieval.py first → {RAW_DATA_PATH}")
  df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
  print(f"Loaded {len(df):,} raw matches")
  print(df.columns)
  
  
