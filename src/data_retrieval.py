import requests
import pandas as pd
from io import StringIO
from tdqm import tqdm
import os

# Configuration constants
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
YEARS = range(2000, 2026)
OUTPUT_PATH = "data/raw_matches.csv"


def fetch_year(year: int) -> pd.DataFrame | None:
    """
    Fetch match data for a specific year.
    
    Args:
        year: The year for which to fetch match data.
    
    Returns:
        A DataFrame containing the match data for the specified year.
    """
    url = f"{BASE_URL}/atp_matches_{year}.csv"
    response = requests.get(url, timeout=15)

    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text), low_memory=False)
    else:
        print(f"Failed to retrieve data for {year}. Status code: {response.status_code}")
        return None
      
def fetch_all_years() -> pd.DataFrame:
  frames = []
  for year in tqdm(YEARS, desc="Fetching ATP matches data"):
    df = fetch_year(year)
    if df is not None:
      df["year"] = year
      frames.append(df)
  return pd.concat(frames, ignore_index=True)
    
  


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = fetch_all_years()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(df):,} matches to {OUTPUT_PATH}")
    print(df.shape)
    print(df.head(3).to_string())