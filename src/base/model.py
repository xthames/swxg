import os

import numpy as np
import pandas as pd
import datetime as dt


class SWXGModel:
    """
    The base class to create, debias, fit, synthesize, and validate the stochastic 
    weather generation model.
    """

    def __init__(self, raw_data: pd.DataFrame) -> None:
        self._raw_data = raw_data
        
        assert len(self._raw_data.columns) >= 4, "Input dataframe must have at least 4 columns!"
        assert ("SITE" in self._raw_data.columns) and (self._raw_data.dtypes["SITE"] is np.dtype('O')), "Location ID column must be labeled 'SITE' with type 'str'!" 


