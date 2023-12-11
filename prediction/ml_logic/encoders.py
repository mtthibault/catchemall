import math
import numpy as np
import pandas as pd
# import pygeohash as gh

from prediction.utils import simple_time_and_memory_tracker

def transform_time_features(X: pd.DataFrame) -> np.ndarray:
    pass

def transform_lonlat_features(X: pd.DataFrame) -> pd.DataFrame:
    pass

def compute_geohash(X: pd.DataFrame, precision: int = 5) -> np.ndarray:
    """
    Add a geohash (ex: "dr5rx") of len "precision" = 5 by default
    corresponding to each (lon, lat) tuple, for pick-up, and drop-off
    """
    pass
