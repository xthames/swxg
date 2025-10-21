import os
import numpy as np
import pandas as pd
import datetime as dt
import warnings

from .fit import fit_data
from .synthesize import synthesize_data


class SWXGModel:
    def __init__(self, raw_data: pd.DataFrame) -> None:
        """
        The base class to create, debias, fit, synthesize, and validate the stochastic 
        weather generation model.

        Parameters
        ----------
        raw_data: pd.DataFrame
            Input dataframe to use as for stochastic weather generation

        Attributes
        ----------
        raw_data: pd.DataFrame
            Mirrors in the input ``raw_data`` parameter
        data: pd.DataFrame
            Temporal reformatting of ``raw_data`` so that each year, month, day have their own column
            in the DataFrame
        resolution: str
            Determined resolution of the observed data. Can be "monthly" or "daily"
        precip_fit_dict: dict
            Dictionary containing statistical information related to fitting of precipitation data
        copulaetemp_fit_dict: dict
            Dictionary containing statistical information related to fitting of copulae and temperature data
        is_fit: bool:
            Flag to confirm that this instance of the generator has been fit
        """
        
        self.raw_data = raw_data
        self.resolution = ""
        self.data = pd.DataFrame()
        self.precip_fit_dict = {}
        self.copulaetemp_fit_dict = {}
        self.is_fit = False

        assert len(self.raw_data.columns) >= 4, "Input dataframe must have at least 4 columns!"
        assert ("SITE" in self.raw_data.columns) and (self.raw_data["SITE"].dtype == object), "Location ID column must be labeled 'SITE' with type 'str'!" 
        assert ("DATETIME" in self.raw_data.columns) and (self.raw_data["DATETIME"].dtype in [np.dtype('<M8[ns]'), np.dtype('>M8[ns]')]), "Date/Time column must be labeled 'DATETIME' with type datetime64[ns]!"
        assert ("PRECIP" in self.raw_data.columns) and (self.raw_data["PRECIP"].dtype == np.dtype('float64')), "Precipitation (with units [m]) column must be labeled 'PRECIP' with type 'float'!"
        assert ("TEMP" in self.raw_data.columns) and (self.raw_data["TEMP"].dtype == np.dtype('float64')), "Temperature (with units [degC]) column must be labeled 'TEMP' with type 'float'!"
        self.format_time_resolution(self.raw_data)


    def format_time_resolution(self, data: pd.DataFrame) -> None:
        """
        Function that separates the raw data's datetime stamps to individual dataframe 
        columns based on the input resolution
        
        Parameters
        ----------
        data: pd.DataFrame
            Input raw data to be used for the fitting
        """
        
        # determine the resolution
        days = [int(data.iloc[i]["DATETIME"].day) for i in range(data.shape[0])]
        resolution = "monthly" if len(set(days)) == 1 else "daily"
        subdaily_flag = False

        # define dataframe columns, datatypes
        if resolution == "monthly":
            stamp_cols = ["SITE", "YEAR", "MONTH", "PRECIP", *data.columns[3:]]
        else:
            stamp_cols = ["SITE", "YEAR", "MONTH", "DAY", "PRECIP", *data.columns[3:]]
        stamp_dtypes = {col: float for col in stamp_cols}
        stamp_dtypes["SITE"] = str
        stamp_dtypes["YEAR"], stamp_dtypes["MONTH"] = int, int
        if resolution == "daily":
            stamp_dtypes["DAY"] = int

        # separate dt.datetime column into years, months, (days)
        wvar_columns = ["PRECIP", *data.columns[3:]]
        dt_stamp_dict, i, subdaily_dict = {}, 0, {wvar: {} for wvar in wvar_columns}
        for i in range(data.shape[0]):
            df_row = data.iloc[i]
            site, year, month = df_row["SITE"], df_row["DATETIME"].year, df_row["DATETIME"].month
            precip = df_row["PRECIP"]
            temp_plus = [df_row[col] for col in data.columns[3:]]
            if resolution == "monthly":
                dt_stamp_dict[i] = [site, year, month, precip, *temp_plus]
            else:
                day = df_row["DATETIME"].day
                wvars = [precip, *temp_plus]
                for wv, wvar_col in enumerate(wvar_columns):
                    if (site, year, month, day) not in subdaily_dict[wvar_col]:
                        subdaily_dict[wvar_col][(site, year, month, day)] = [wvars[wv]] 
                    else:
                        subdaily_dict[wvar_col][(site, year, month, day)].append(wvars[wv])
                        subdaily_flag = True
                if not subdaily_flag:
                    dt_stamp_dict[i] = [site, year, month, day, precip, *temp_plus] 
        # if subdaily, aggregate/average to daily
        if subdaily_flag:
            for i, k in enumerate(subdaily_dict["PRECIP"].keys()):
                wvars = []
                for wvar_col in wvar_columns:
                    if wvar_col == "PRECIP": 
                        wvars.append(np.nansum(subdaily_dict[wvar_col][k]))
                    else:
                        wvars.append(np.nanmean(subdaily_dict[wvar_col][k]))
                dt_stamp_dict[i] = [*k, *wvars] 
        dt_stamp_df = pd.DataFrame().from_dict(dt_stamp_dict, orient="index", columns=stamp_cols)
        dt_stamp_df.reset_index(drop=True, inplace=True)
        dt_stamp_df.astype(stamp_dtypes) 

        self.data, self.resolution = dt_stamp_df, resolution
        if subdaily_flag:
            warnings.warn("Subdaily data resolution detected! Subdaily synthesizing not yet implemented, aggregated to daily...", UserWarning)
        if len(set(self.data["YEAR"].values)) < 20:
            warnings.warn("Fewer than 20 years of data detected! Fit and synthesizing is possible, but carefully review fit validation before synthesizing...", UserWarning)

    
    def fit(self, 
            verbose: bool = True,
            validate: bool = True,
            dirpath: str = "",
            kwargs: dict = {}) -> None:
        """
        Local method that calls the ``fit_data`` function, which aptly fits the input 
        weather data by returning extracted statistical parameters and reformatted data 

        Parameters
        ----------
        verbose: bool, optional
            Flag for displaying precipitation and temperature fit statistics. Default: True
        validate: bool, optional
            Flag for producing figures to validate each step of the generator. Default: True
        dirpath: str, optional
            Path for where to save the validation figures. Default: ""
        kwargs: dict, optional
            Keyword arguments related to the fit. Leaving this empty sets the keyword
            arguments to their default values. Keywords are:
            
             * ``gmmhmm_min_states``: int, default = 1
             * ``gmmhmm_max_states``: int, default = 4
             * ``gmmhmm_states``: int, default = 0
             * ``ar_lag``: int, default = 1
             * ``stationarity_groups``: int, default = 2
             * ``copula_families``: list[str], default = ["Independence", "Frank", "Gaussian"]
             * ``figure_extension``: str, default="svg"
             * ``validation_figures``: list[str], default = ["precip", "copula"]
        """
        
        kwargs["fit_verbose"] = verbose
        self.precip_fit_dict, self.copulaetemp_fit_dict = fit_data(self.data, self.resolution, validate, dirpath, kwargs)
        self.is_fit = True
        
        if verbose:
            print("--------------- Precipitation Fit ---------------")
            verbose_precip_sites, verbose_precip_cols = list(self.precip_fit_dict["log10_annual_precip"].columns), ["STATE", "SITE", "MEANS", "STDS"]
            verbose_gmmhmm_df = pd.DataFrame(columns=verbose_precip_cols)
            states, sites, means, stds = [], [], [], []
            for s in range(self.precip_fit_dict["num_gmmhmm_states"]):
                states.extend([s] * len(verbose_precip_sites))
                for i, site in enumerate(verbose_precip_sites):
                    sites.append(site)
                    means.append(self.precip_fit_dict["means"][s][i])
                    stds.append(self.precip_fit_dict["stds"][s][i])
            verbose_gmmhmm_df["STATE"], verbose_gmmhmm_df["SITE"], verbose_gmmhmm_df["MEANS"], verbose_gmmhmm_df["STDS"] = states, sites, means, stds
            verbose_transprob_idxs = ["FROM STATE {}".format(t) for t in range(self.precip_fit_dict["num_gmmhmm_states"])]
            verbose_transprob_cols = ["TO STATE {}".format(t) for t in range(self.precip_fit_dict["num_gmmhmm_states"])]
            verbose_transprob_df = pd.DataFrame(index=verbose_transprob_idxs, columns=verbose_transprob_cols) 
            for j in range(self.precip_fit_dict["num_gmmhmm_states"]):
                for k in range(self.precip_fit_dict["num_gmmhmm_states"]):
                    verbose_transprob_df.at["FROM STATE {}".format(j), "TO STATE {}".format(k)] = self.precip_fit_dict["t_probs"][j][k]
            print("* Number of GMMHMM States: {}".format(self.precip_fit_dict["num_gmmhmm_states"])) 
            print(" ")
            print("* GMMHMM Means/Stds per Site and State")
            print(verbose_gmmhmm_df.to_string(index=False))
            print(" ")
            print("* Transition Probability Matrix")
            print(verbose_transprob_df.to_string())
            print("-------------------------------------------------") 
            print(" ")
            print("------------------ Copulas Fit ------------------")
            months = self.copulaetemp_fit_dict.keys()
            for month in months:
                month_df = self.copulaetemp_fit_dict[month]
                print("Copula Statistics for: {}".format(month.upper()))
                print("* Best-Fitting Copula Family: {}".format(month_df["BestCopula"][1]))
                print("* All Family Parameters and Fit Comparison")
                families_df = month_df["CopulaDF"]
                families_df.drop("Copula", axis=1, inplace=True)
                families_df.rename(columns={"params": "Hyperparameter", "S_n": "Cram\u00e9r von Mises", "T_n": "Kolmogorov-Smirnov"}, inplace=True)
                print(families_df.to_string())
                print(" ")
            print("-------------------------------------------------") 

    
    def synthesize(self,
                   n: int = 0,
                   resolution: str = "",
                   validate: bool = True,
                   dirpath: str = "",
                   kwargs: dict = {}) -> pd.DataFrame:
        """
        Local method that calls the ``synthesize_data`` function, which aptly 
        synthesizes weather through either observed fit or given statistical parameters 

        Parameters
        ----------
        n: int, optional
            Number of years to synthesize. Default takes the same size as the number of
            years in the observed dataset
        resolution: str, optional
            The resolution to synthesize the data at. Leaving this empty sets the same
            resolution as the raw data
        validate: bool, optional
            Flag for producing figures to validate each step of the generator. Default: True
        dirpath: str, optional
            Path for where to save the validation figures. Default: ""
        kwargs: dict, optional
            Keyword arguments related to the fit. Leaving this empty sets the keyword
            arguments to their default values. Keywords are:
            
             * ``validation_samplesize_mult``: int, default = 10
             * ``figure_extension``: str, default="svg"

        Returns
        -------
        synthesized_data: pd.DataFrame
            The synthesized weather data at the desired time resolution
        """

        assert not self.data.empty, "Must include a dataframe of weather observations but none found!"
        assert self.is_fit, "Data has not been fit and therefore cannot synthesize!" 
        resolution = self.resolution if resolution == "" else resolution

        n = len(set(self.data["YEAR"].values)) if n <= 0 else n
        if ("DAY" in self.data) and (resolution == "monthly"):
            obs_dict = {}
            for site in sorted(set(self.data["SITE"].values)):
                site_idx = self.data["SITE"] == site
                site_entry = self.data.loc[site_idx]
                for year in sorted(set(site_entry["YEAR"].values)):
                    year_idx = site_entry["YEAR"] == year
                    year_entry = site_entry.loc[year_idx]
                    for month in sorted(set(year_entry["MONTH"].values)):
                        month_idx = year_entry["MONTH"] == month
                        month_entry = year_entry.loc[month_idx]
                        prcps, temps = month_entry["PRECIP"].values, month_entry["TEMP"].values
                        obs_dict[(site, year, month)] = [site, year, month, np.nansum(prcps), np.nanmean(temps)]
            obs_data = pd.DataFrame().from_dict(obs_dict, orient="index", columns=["SITE", "YEAR", "MONTH", "PRECIP", "TEMP"])
            obs_data.reset_index(drop=True, inplace=True)
            obs_data.astype({"SITE": str, "YEAR": int, "MONTH": int, "PRECIP": float, "TEMP": float})
        else:
            obs_data = self.data

        synthesized_data = synthesize_data(n, obs_data, self.precip_fit_dict, self.copulaetemp_fit_dict, 
                                           resolution, validate, dirpath, kwargs) 
        return synthesized_data
