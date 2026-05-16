import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
  ConfusionMatrixDisplay,
  RocCurveDisplay,
  classification_report,
  roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import joblib

warnings.filterwarnings("ignore")

#### ----------------- Constants -----------------
PLOTS_PATH = "plots"
MODELS_PATH = "models"
DATA_DIR = "data"
MATCHES_CSV = os.path.join(DATA_DIR,"matches_data.csv")
PLAYERS_CSV = os.path.join(DATA_DIR,"players_data.csv")



#### --------------- Features Sets ---------------
# For the comeback model we use winner-perspective stats.
# For the upset model we use delta features (winner − loser).
COMEBACK_FEATURES = [
  "w_ace_rate", "w_df_rate", "w_1stIn_pct", "w_1stWon_pct", 
  "w_2ndWon_pct", "w_bpSaved_pct", "w_srvWon_pct", "w_retWon_pct",
  "l_ace_rate", "l_df_rate", "l_1stIn_pct", "l_1stWon_pct", "l_bpSaved_pct",
  "rank_gap", 
  "best_of_5",
  "surf_hard", "surf_clay", "surf_grass", "surf_carpet",
  "is_gs", "is_masters", "is_atp", "is_finals", "is_davis"
]

UPSET_FEATURES = [
  "delta_ace_pct", "delta_1stWon_pct", "delta_bpSaved_pct", 
  "delta_srvWon_pct", "delta_retWon_pct",
  "rank_gap",
  "best_of_5",
  "surf_hard", "surf_clay", "surf_grass", "surf_carpet",
  "is_gs", "is_masters", "is_atp", "is_finals", "is_davis"
]



#### --------------- Colors Tokens ---------------
DARK      = "#0D1117"   # Figure BACKGROUND
PANEL     = "#161B22"   # Axes BACKGROUND
GRID      = "#30363D"   # Grid LINES
TEXT      = "#E6EDF3"   # Primary LABELS 
SUBTEXT   = "#8B949E"   # Tick LABELS
CLAY      = "#C97B4B"
GRASS     = "#4CAF50"
HARD      = "#1E88E5"
CARPET    = "#9C27B0"
GOLD      = "#F0C040"
RED       = "#CF6679"
GREEN     = "#4CAF50"
BLUE      = "#1E88E5"
SURF_COLOR = {"Clay": CLAY, "Grass": GRASS, "Hard": HARD, "Carpet": CARPET}



#### ------------------ Helper ------------------
def _apply_theme() -> None:
  """
  Set global matplotlib rcParams once. Called at module import time.
  
  Returns:
    None
  """
  plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linestyle":    "--",
    "grid.alpha":        0.45,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.framealpha": 0,
    "legend.labelcolor": TEXT,
    "legend.fontsize":   8,
  })

_apply_theme()


#### ------------- Training Pipeline -------------
def _build_models(pos_weight: float) -> dict:
  """Return a dict of named sklearn-compatible model objects. All are wrapped in a Pipeline with StandardScaler. StandardScaler is a no-op for tree models in terms of accuracy,
  but keeps the API uniform
  
  Args:
    pos_weight (float): Ratio of negative to positive samples passed to XGBoost to handle class imbalance without resampling.
  
  Returns:
    dict: A dictionary of sklearn model objects wrapped in a Pipeline
  """
  return {
    "Logistic Regression": Pipeline([
      ("scaler", StandardScaler()),
      ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced", # auto-adjusts class weights
        l1_ratio=0, # "0" = L2 regularization applied
        C=1.0, # inverse regularization strength
        solver="lbfgs", # optimization algorithm
        random_state=42
      ))
    ]),
    "Random Forest": Pipeline([
      ("scaler", StandardScaler()),
      ("clf", RandomForestClassifier(
        n_estimators=400, # number of trees
        max_depth=8, # maximum depth of a tree
        min_samples_leaf=20, # min samples for a leaf
        class_weight="balanced_subsample", # auto-adjusts class weights from boostrap sample
        n_jobs=-1,
        random_state=42
      ))
    ]),
    "XGBoost": Pipeline([
      ("scaler", StandardScaler()),
      ("clf", XGBClassifier(
        n_estimators=400, # number of trees
        learning_rate=0.05, # step size [0,1] 
        max_depth=6, # maximum depth of a tree
        subsample=0.8, # subsample ratio of the training instances
        colsample_bytree=0.8, # fraction of columns to be subsampled for each tree
        scale_pos_weight=pos_weight, # control balance of classes, unbalance = (num_neg / num_pos)
        eval_metric="logloss", # negative log-likelihood loss
        use_label_encoder=False,
        verbosity=0,
        random_state=42
      ))
    ]),
  }

def train_task(
  df: pd.DataFrame,
  feature_cols: list[str],
  target_col: str,
  task_name: str,
) -> dict:
  """Full training pipeline for one classification task.

  Args:
    df (pd.DataFrame): Dataset used to trian the model
    feature_cols (list[str]): List of string of the columns of the dataset to use as features
    target_col (str): Name of the column of the dataset to use as the target
  
  Returns:
    dict: A dictionary with the model, feature_cols, test accuracy and the task name  
  pass
  """
  print(f"\n{"═"*60}")
  print(f"  TASK: {task_name}")
  print(f"{"═"*60}")
  
  # Prepare data
  available = [c for c in feature_cols if c in df.columns]
  sub = df[available + [target_col]].dropna()
  X = sub[available].values
  y = sub[target_col].values.astype(int)
  print(f"Samples: {len(X):,}")
  print(f"Positive Rate: {y.mean():.2%}")
  print(f"Features: {len(available)}")
  
  # Train-test split
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
  )
  
  # Cross-validation
  neg_to_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
  models = _build_models(neg_to_pos_ratio)
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  cv_results = {}
  print(f"\n{"Model":<25}  CV AUC (mean ± std)")
  print(f"{"-"*45}")
  for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores.mean()
    print(f"{name:<25}  {scores.mean():.4f} ± {scores.std():.4f}")
  
  # Refit best model
  best_name = max(cv_results, key=cv_results.get)
  best_model = models[best_name]
  best_model.fit(X_train, y_train)
  print(f"\nBest model: {best_name}")
  
  # Test set evaluation
  y_pred = best_model.predict(X_test)
  y_prob = best_model.predict_proba(X_test)[:,1]
  auc = roc_auc_score(y_test, y_prob)
  print(f"\nTest AUC: {auc:.4f}\n")
  print(classification_report(y_test, y_pred, target_names=["No", "Yes"], digits=3))
  
  # Save evaluation plots (ROC Curve & Confusion Matrix)
  tag = task_name.lower().replace(" ", "_")
  fig, axes = plt.subplots(1, 2, figsize=(14,5))
  fig.suptitle(f"{task_name} — {best_name}  (Test AUC: {auc:.3f})",
               color=TEXT, fontsize=12, weight="bold")
  
  RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0], color=GOLD, name=best_name)
  axes[0].plot([0,1],[0,1], color=GRID, linestyle="--")
  axes[0].set_title("ROC Curve")
  
  ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[1], colorbar=False, cmap="YlOrBr")
  axes[1].set_title("Confusion Matrix")
  
  plt.tight_layout()
  path = f"plots/{tag}_evaluation.png"
  plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK, edgecolor="none")
  plt.close()
  print(f"\nSaved Evaluation Plots: \"plots/{tag}_evaluation.png\"")
  
  # Persit model
  artifact = {
    "model":        best_model,
    "feature_cols": available,
    "task_name":    task_name,
    "test_auc":     auc,
    "best_cv_name": best_name,
  }
  model_path = f"models/{tag}.pkl"
  joblib.dump(artifact, model_path)
  print(f"\nSaved Model: \"{model_path}\"")
  return artifact
  
#### ------------------- Main -------------------
if __name__ == "__main__":
  os.makedirs("models", exist_ok=True)
  os.makedirs("plots",  exist_ok=True)
  if not os.path.exists(MATCHES_CSV):
    raise FileNotFoundError("Run feature_engineering.py first.")

  df = pd.read_csv(MATCHES_CSV, low_memory=False)
  print(f"Loaded {len(df):,} matches")

  # Task 1: Comeback classifier
  train_task(
    df            = df[df["came_back"].notna()],
    feature_cols  = COMEBACK_FEATURES,
    target_col    = "came_back",
    task_name     = "Comeback Classifier",
  )

  # Task 2: Upset predictor
  train_task(
    df            = df[df["upset"].notna()],
    feature_cols  = UPSET_FEATURES,
    target_col    = "upset",
    task_name     = "Upset Predictor",
  )

  print("\nAll models trained and saved to models/")