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
        self.raw_data: pd.DataFrame = raw_data
        self.data: pd.DataFrame = pd.DataFrame()
        self.precip_fit_dict: dict = {}

        assert len(self.raw_data.columns) >= 4, "Input dataframe must have at least 4 columns!"
        assert ("SITE" in self.raw_data.columns) and (self.raw_data.dtypes["SITE"] is np.dtype('O')), "Location ID column must be labeled 'SITE' with type 'str'!" 
        assert ("DATETIME" in self.raw_data.columns) and (self.raw_data.dtypes["DATETIME"] in [np.dtype('<M8[ns]'), np.dtype('>M8[ns]')]), "Date/Time column must be labeled 'DATETIME' with type datetime64[ns]!"
        assert ("PRECIP" in self.raw_data.columns) and (self.raw_data.dtypes["PRECIP"] is np.dtype('float64')), "Precipitation column must be labeled 'PRECIP' with type 'float'!"
        assert np.all(np.all(~np.isnan(self.raw_data[self.raw_data.columns[2:]].values))), "Missing data/NaNs detected -- fill with appropriate averages or remove!"

        data, precip_fit_dict = self.fit()
        self.data = data
        self.precip_fit_dict = precip_fit_dict


    def fit(self, resolution: str = "daily") -> float:
        return fit_data(self.raw_data, resolution)

