import requests
import pandas as pd
from datetime import datetime
from io import StringIO
from tqdm import tqdm
import os

# Configuration constants
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
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
      
def fetch_all_years(years: list[int]) -> pd.DataFrame:
  """
  Fetch match data for all years defined in the YEARS constant.
  
  Returns:
    A DataFrame containing the match data for all specified years.
  """
  frames = []
  for year in tqdm(years, desc="Fetching ATP matches data"):
    df = fetch_year(year)
    if df is not None:
      df["year"] = year
      frames.append(df)
  return pd.concat(frames, ignore_index=True)
    
if __name__ == "__main__":
  os.makedirs("data", exist_ok=True)
  
  current_year = datetime.now().year
  years = range(2000, current_year + 1)
  df = fetch_all_years(years=years)
  df.to_csv(OUTPUT_PATH, index=False)
  
  print(f"\nSaved {len(df):,} matches to {OUTPUT_PATH}")
  print(df.shape)
  print(df.head(3).to_string())