import numpy as np
import pandas as pd
import datetime as dt

from .fit import *


class SWXGModel:
    """
    The base class to create, debias, fit, synthesize, and validate the stochastic 
    weather generation model.
    """

    def __init__(self, raw_data: pd.DataFrame) -> None:
        self.raw_data = raw_data

        assert len(self.raw_data.columns) >= 4, "Input dataframe must have at least 4 columns!"
        assert ("SITE" in self.raw_data.columns) and (self.raw_data.dtypes["SITE"] is np.dtype('O')), "Location ID column must be labeled 'SITE' with type 'str'!" 
        assert ("DATETIME" in self.raw_data.columns) and (self.raw_data.dtypes["DATETIME"] in [np.dtype('<M8[ns]'), np.dtype('>M8[ns]')]), "Date/Time column must be labeled 'DATETIME' with type datetime64[ns]!"
        assert ("PRECIP" in self.raw_data.columns) and (self.raw_data.dtypes["PRECIP"] is np.dtype('float64')), "Precipitation column must be labeled 'PRECIP' with type 'float'!"

    def fit(self, resolution: str = "annual") -> float:
        return fit_data(self.raw_data, resolution)
