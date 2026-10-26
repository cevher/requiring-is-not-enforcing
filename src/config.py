"""
Project configuration.

2005 revision:
- Main empirical window starts in 2005 to capture the 2008 global financial crisis
  and increase the number of temporal validation snapshots.
- Firm availability is handled dynamically at the rolling-window level.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

START_DATE = "2005-01-01"
END_DATE = "2025-12-31"

UNIVERSE_PATH = PROJECT_ROOT / "data_raw" / "metadata" / "universe.csv"

PRICE_RAW_DIR = PROJECT_ROOT / "data_raw" / "prices"
MACRO_RAW_DIR = PROJECT_ROOT / "data_raw" / "macro"

RETURNS_DIR = PROJECT_ROOT / "data_processed" / "returns"
NETWORK_DIR = PROJECT_ROOT / "data_processed" / "networks"
LABEL_DIR = PROJECT_ROOT / "data_processed" / "labels"
EVIDENCE_DIR = PROJECT_ROOT / "data_processed" / "evidence"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

# Rolling network parameters
ROLLING_WINDOW_DAYS = 60
CORR_THRESHOLD = 0.50
TOP_K_NEIGHBORS = 5

# Dynamic universe/window-level filtering
# A ticker does not need to exist for the whole 2005-2025 period.
# It only needs sufficient observations inside the current rolling window.
MIN_WINDOW_OBS_RATIO = 0.80
MIN_TICKERS_PER_WINDOW = 20

# Stress label parameters
VIX_HIGH_QUANTILE = 0.80
RETURN_STRESS_QUANTILE = 0.10
MIN_MARKET_TICKERS_PER_DAY = 20
