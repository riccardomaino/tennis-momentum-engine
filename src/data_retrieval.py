import requests
import pandas as pd
from datetime import datetime
from io import StringIO
from tqdm import tqdm
import os

# Configuration constants
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
CURRENT_YEAR = datetime.now().year
YEARS = range(2000, CURRENT_YEAR+1)
OUTPUT_DIR = "data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "raw_matches.csv")


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
  try:
    if response.status_code == 200:
      df = pd.read_csv(StringIO(response.text), low_memory=False)
      df["year"] = year
      return df
    else:
      return None
  except requests.RequestException as exc:
    print(f"Network error for {year}: {exc}")
    return None
      
def fetch_all_years(years: list[int] = YEARS) -> pd.DataFrame:
  """
  Fetch match data for all years defined in the YEARS constant.
  
  Returns:
    A DataFrame containing the match data for all specified years.
  """
  frames = []
  for year in tqdm(years, desc="Downloading ATP data", unit="year"):
    df = fetch_year(year)
    if df is not None:
      frames.append(df)
  master = pd.concat(frames, ignore_index=True)
  return master
    
if __name__ == "__main__":
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  
  print(f"Fetching ATP match data ({min(YEARS)} - {max(YEARS)})…\n")
  df = fetch_all_years(years=YEARS)
  df.to_csv(OUTPUT_CSV, index=False)
  
  print(f"\nSaved {len(df):,} matches → {OUTPUT_CSV}")
  print(f"    Columns : {list(df.columns[:10])} ... ({df.shape[1]} total)")
  print(f"    Years   : {df["year"].min()} - {df["year"].max()}")
  print(f"    Memory  : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")