from src.base.model import SWXGModel

import numpy as np
import pandas as pd


def test_base() -> None:
    
    df: pd.DataFrame = pd.DataFrame({"STATION": ["a", "b", "c"], 
                                     "YEAR": [2025, 2025, 2025],
                                     "MONTH": [1, 1, 1],
                                     "DAY": [1, 2, 3],
                                     "PRECIP": [0., 0.5, 1.0],
                                     "TEMP": [15., 14., 16.]})

    model = SWXGModel(df)
    assert model.raw_data.equals(df)

