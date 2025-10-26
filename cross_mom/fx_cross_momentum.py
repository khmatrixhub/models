"""
Cross-sectional momentum strategy for intraday FX trading.

Strategy:
- Every X minutes, rank currency pairs by momentum relative to USD
- Go long top N pairs, short bottom N pairs (USD neutral)
- Intraday trading session: 5PM NY to 5PM NY next day
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Callable
import glob

# ...existing code...
