"""The frame units of the study (data/units.csv): unit (s, k) = real frame k of FaceForensics++ video s;
t = the video whose scene FaceForensics++ used with the face of s (fake t_s_fk)."""
import os
from functools import lru_cache
import pandas as pd
from common import DATA_DIR


@lru_cache(None)
def units():
    return pd.read_csv(os.path.join(DATA_DIR, "units.csv"))
