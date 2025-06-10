import numpy as np
import pandas as pd
import datetime as dt

from .fit import *


class SWXGModel:
    def __init__(self, raw_data: pd.DataFrame) -> None:
        """
        The base class to create, debias, fit, synthesize, and validate the stochastic 
        weather generation model.

        Parameters
        ----------
        raw_data: pd.DataFrame
            Input dataframe to use as for stochastic weather generation

        Properties
        ----------
        raw_data: pd.DataFrame
            Mirrors in the input ``raw_data`` parameter
        data: pd.DataFrame
            Temporal reformatting of ``raw_data`` so that each year, month, day have their own column
            in the DataFrame
        precip_fit_dict: dict
            Dictionary containing statistical information related to fitting of precipitation data
        """
        
        self.raw_data = raw_data
        self.data = pd.DataFrame()
        self.precip_fit_dict = {}

        assert len(self.raw_data.columns) >= 4, "Input dataframe must have at least 4 columns!"
        assert ("SITE" in self.raw_data.columns) and (self.raw_data.dtypes["SITE"] is np.dtype('O')), "Location ID column must be labeled 'SITE' with type 'str'!" 
        assert ("DATETIME" in self.raw_data.columns) and (self.raw_data.dtypes["DATETIME"] in [np.dtype('<M8[ns]'), np.dtype('>M8[ns]')]), "Date/Time column must be labeled 'DATETIME' with type datetime64[ns]!"
        assert ("PRECIP" in self.raw_data.columns) and (self.raw_data.dtypes["PRECIP"] is np.dtype('float64')), "Precipitation column must be labeled 'PRECIP' with type 'float'!"
        assert np.all(np.all(~np.isnan(self.raw_data[self.raw_data.columns[2:]].values))), "Missing data/NaNs detected -- fill entries or remove from dataframe!"


    def fit(self, resolution: str = "daily") -> None:
        """
        Local method that calls the ``fit_data`` function, which aptly fits the input 
        weather data by returning extracted statistical parameters and reformatted data 

        Parameters
        ----------
        resolution: str, optional
            The temporal resolution of the input data. Can be 'monthly' or 'daily'. Default: 'daily' 
        """
        self.data, self.precip_fit_dict = fit_data(self.raw_data, resolution)

