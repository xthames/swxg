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
        self.copulaetemp_fit_dict = {}

        assert len(self.raw_data.columns) >= 4, "Input dataframe must have at least 4 columns!"
        assert ("SITE" in self.raw_data.columns) and (self.raw_data.dtypes["SITE"] is np.dtype('O')), "Location ID column must be labeled 'SITE' with type 'str'!" 
        assert ("DATETIME" in self.raw_data.columns) and (self.raw_data.dtypes["DATETIME"] in [np.dtype('<M8[ns]'), np.dtype('>M8[ns]')]), "Date/Time column must be labeled 'DATETIME' with type datetime64[ns]!"
        assert ("PRECIP" in self.raw_data.columns) and (self.raw_data.dtypes["PRECIP"] is np.dtype('float64')), "Precipitation column must be labeled 'PRECIP' with type 'float'!"
        assert ("TEMP" in self.raw_data.columns) and (self.raw_data.dtypes["TEMP"] is np.dtype('float64')), "Temperature column must be labeled 'TEMP' with type 'float'!"
        assert np.all(np.all(~np.isnan(self.raw_data[self.raw_data.columns[2:]].values))), "Missing data/NaNs detected -- fill entries or remove from dataframe!"


    def fit(self, 
            resolution: str = "daily",
            validate: bool = True,
            dirpath: str = "",
            fit_kwargs: dict = {}) -> None:
        """
        Local method that calls the ``fit_data`` function, which aptly fits the input 
        weather data by returning extracted statistical parameters and reformatted data 

        Parameters
        ----------
        resolution: str, optional
            The temporal resolution of the input data. Can be 'monthly' or 'daily'. Default: 'daily' 
        validate: bool, optional
            Flag for producing figures to validate each step of the generator. Default: True
        dirpath: str, optional
            Path for where to save the validation figures. Default: ""
        fit_kwargs: dict, optional
            Keyword arguments related to the fit. Leaving this empty sets the keyword
            arguments to their default values. Keywords are:
            ``gmmhmm_min_states``: int, default = 1
            ``gmmhmm_max_states``: int, default = 4
            ``ar_lag``: int, default = 1
            ``stationarity_groups``: int, default = 2
            ``copula_families: list[str], default = ["Frank"]
        """
        self.data, self.precip_fit_dict, self.copulaetemp_fit_dict = fit_data(self.raw_data, resolution, validate, dirpath, fit_kwargs)

