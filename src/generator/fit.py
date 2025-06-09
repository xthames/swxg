from typing import List, Tuple, Dict 
import numpy as np
import pandas as pd
import datetime as dt



def fit_data(raw_data: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """
    Managing function that fits the raw climate/weather data as a reformatted DataFrame,
    with statistics and sampling schemes for precipitation, additional parameters
    (i.e. temperature), and the relationship between the two
    """
    
    formatted_data: pd.DataFrame = format_time_resolution(raw_data, resolution)
    precip_col_idx = list(formatted_data.columns).index("PRECIP")
    precip_fit_dict: Dict = fit_precip(formatted_data[formatted_data.columns[:precip_col_idx+1]].copy(), resolution)

    return formatted_data, precip_fit_dict


def format_time_resolution(data: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """
    Function that separates the raw data's datetime stamps to individual dataframe 
    columns based on the input resolution
    """
    
    assert resolution in ["monthly", "daily"], "Generator resolution can only be 'monthly' or 'daily'!"
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
    dt_stamp_dict = {}
    for i in range(data.shape[0]):
        df_row = data.iloc[i]
        site, year, month = df_row["SITE"], df_row["DATETIME"].year, df_row["DATETIME"].month
        precip = df_row["PRECIP"]
        temp_plus = [df_row[col] for col in data.columns[3:]]
        if resolution == "monthly":
            dt_stamp_dict[i] = [site, year, month, precip, *temp_plus]
        else:
            day = df_row["DATETIME"].day
            dt_stamp_dict[i] = [site, year, month, day, precip, *temp_plus]
    dt_stamp_df = pd.DataFrame().from_dict(dt_stamp_dict, orient="index", columns=stamp_cols)
    dt_stamp_df.reset_index(drop=True, inplace=True)
    dt_stamp_df.astype(stamp_dtypes) 

    return dt_stamp_df


def fit_precip(data: pd.DataFrame, resolution: str) -> dict:
    """
    Function that fits and validates the precipitation data. Precipitation is transformed
    to a log-scale, annualized (summed), and fit to a Gaussian mixture-model Hidden 
    Markov model (GMMHMM)
    """

    # annualize precipitation, log10 transformation
    transformed_dict = {}
    for site in sorted(set(data["SITE"].values)):
        site_idx = data["SITE"] == site
        site_entry = data[site_idx]
        site_years = sorted(set(site_entry["YEAR"].values))
        for year in sorted(set(site_entry["YEAR"].values)):
            year_idx = site_entry["YEAR"] == year
            year_entry = site_entry[year_idx]
            annualized_precip = np.log10(np.sum(year_entry["PRECIP"].values))
            transformed_dict[(site, year)] = [site, year, annualized_precip]
    transformed_data = pd.DataFrame().from_dict(transformed_dict, orient="index", columns=["SITE", "YEAR", "PRECIP"])
    transformed_data.reset_index(drop=True, inplace=True)
    transformed_data.astype({"SITE": str, "YEAR": int, "PRECIP": float})

    # determine best-fitting number of states for GMMHMM
    return {"transformed_precip": transformed_data}


