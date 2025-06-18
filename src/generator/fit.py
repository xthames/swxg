from typing import List, Tuple, Dict 
import numpy as np
import pandas as pd
import datetime as dt
import random
from hmmlearn.hmm import GMMHMM

from .validate import *


validation = False


def fit_data(raw_data: pd.DataFrame, resolution: str) -> list[pd.DataFrame, Dict]:
    """
    Managing function that fits the raw climate/weather data as a reformatted DataFrame,
    with statistics and sampling schemes for precipitation, additional parameters
    (i.e. temperature), and the relationship between the two

    Parameters
    ----------
    raw_data: pd.DataFrame
        Input raw data to be used for the fitting
    resolution: str, optional
        The temporal resolution of the input data. Can be 'monthly' or 'daily'. Default: 'daily' 
    
    Returns
    -------
    formatted_data: pd.DataFrame
        Temporal reformatting of ``raw_data`` so that each year, month, day have their own column
        in the DataFrame
    precip_fit_dict: dict
        Dictionary containing statistical information related to fitting of precipitation data
    """
    
    # format
    formatted_data = format_time_resolution(raw_data, resolution)
    # validation
    global validation
    validation = True
    # precip
    precip_col_idx = list(formatted_data.columns).index("PRECIP")
    precip_fit_dict = fit_precip(formatted_data[formatted_data.columns[:precip_col_idx+1]].copy(), resolution)

    return formatted_data, precip_fit_dict


def format_time_resolution(data: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """
    Function that separates the raw data's datetime stamps to individual dataframe 
    columns based on the input resolution
    
    Parameters
    ----------
    data: pd.DataFrame
        Input raw data to be used for the fitting
    resolution: str, optional
        The temporal resolution of the input data. Can be 'monthly' or 'daily'. Default: 'daily' 
    
    Returns
    -------
    dt_stamp_df: pd.DataFrame
        Temporal reformatting of ``data`` so that each year, month, day have their own column
        in the DataFrame
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
    
    Parameters
    ----------
    data: pd.DataFrame
        Temporal reformat of ``raw_data`` where each year, month, day have their own column
        in the DataFrame
    resolution: str, optional
        The temporal resolution of the input data. Can be 'monthly' or 'daily'. Default: 'daily' 
    
    Returns
    -------
    precip_fit_dict: dict
        Dictionary containing statistical information related to fitting of precipitation data
    """

    # annualize precipitation, log10 transformation, format so index=years, column=sites, cell=precip
    sites, years = sorted(set(data["SITE"].values)), sorted(set(data["YEAR"].values))
    transformed_data = pd.DataFrame(index=years, columns=sites)
    for site in sites:
        site_idx = data["SITE"] == site
        site_entry = data[site_idx]
        site_years = sorted(set(site_entry["YEAR"].values))
        for year in years:
            year_idx = site_entry["YEAR"] == year
            year_entry = site_entry[year_idx]
            if year_entry.empty: continue
            transformed_data.at[year, site] = np.log10(np.sum(year_entry["PRECIP"].values))
    transformed_data.astype({col: float for col in sites})
    transformed_data.dropna(inplace=True)

    # checking if there are any years missing from the data, making a sequence existing years
    seq_lengths, l = [], 0
    for year in years:
        if year in list(transformed_data.index):
            l += 1
        else:
            seq_lengths.append(l)
            l = 0
    seq_lengths.append(l)

    # determine best-fitting number of states for GMMHMM
    num_gmmhmm_states = gmmhmm_state_num_estimator(transformed_data, seq_lengths, max_states=4)

    
    return {"transformed_precip": transformed_data,
            "seq_lengths": seq_lengths,
            "num_gmmhmm_states": num_gmmhmm_states}


def gmmhmm_state_num_estimator(transformed_df: pd.DataFrame, lengths: list[int], min_states: int = 1, max_states: int = 5, iterations: int = 10) -> int:
    """
    Function to programmatically determine the best-fitting number of states 
    for the Gaussian mixture model hidden Markov model
    
    Parameters
    ----------
    transformed_df: pd.DataFrame
        DataFrame for the log10-transformed annualized precipitation values, with
        associated sites and years organized as a DataFrame where indices are
        years and columns are sites
    lengths: list[int]
        Length of each sequence of consecutive years in the data
    min_states: int, optional
        The minimum number of hidden states to try fitting. Default: 1 
    max_states: int, optional
        The maximum number of hidden states to try fitting. More than ~6 tends
        to perform poorly in terms of best fit and length of computation. Default: 5 
    iterations: int, optional
        The number of attempts to find a best-fitting model for each unique
        number of states -- this is necessary because the convergence EM method can 
        fall into local optima. From all iterations, the one with the highest 
        log-likelihood is used. Default: 10

    Returns
    -------
    best_num_states: int
        Best-fitting model's number of states, as determined by Bayesian Information
        Criterion (BIC). Log-likelihood and Akaike Information Criterion (AIC) are 
        also used, but log-likelihood is monotonically increasing with number of states
        and (from experimentation) AIC does not penalize additional states strongly enough. 
    """

    n_stations = len(transformed_df.columns)
    models, seeds, AICs, BICs, LLs = [], [], [], [], []
    max_attempts = 50
    for num_states in range(min_states, max_states + 1):
        best_model, best_LL, best_seed = None, None, None
        for _ in range(iterations):
            positive_definite, temp_seed, temp_model = False, None, None
            attempts = 0
            while (not positive_definite) and (attempts < max_attempts):
                # get the random state
                tempSeed = random.getstate()

                # define the parameters for the model
                temp_model_inst = GMMHMM(n_components=num_states, n_iter=1000, covariance_type="full", init_params="cmw")
                temp_model_inst.startprob_ = np.full(shape=num_states, fill_value=1./num_states)
                temp_model_inst.transmat_ = np.full(shape=(num_states, num_states), fill_value=1./num_states)
                temp_model = temp_model_inst.fit(transformed_df.values, lengths=lengths)

                # get, reshape the covariance matrices
                temp_covars = temp_model.covars_.reshape((num_states, n_stations, n_stations))

                # check that each state has a covariance matrix that is positive definite, meaning:
                # -- it's symmetric
                symmetric_check = all([np.allclose(temp_covars[i].T, temp_covars[i]) for i in range(num_states)])
                # -- it's eigenvalues are all positive
                eig_check = all([(np.linalg.eigvalsh(temp_covars[i]) > 0).all() for i in range(num_states)])

                # state if the covariance matrix is positive definite or not
                positive_definite = symmetric_check and eig_check
                attempts += 1

            # calculate the loglikelihood of this model, save best
            if attempts < max_attempts: 
                temp_score = temp_model.score(transformed_df.values, lengths=lengths)
                if not best_LL or temp_score > best_LL:
                    best_LL = temp_score
                    best_model = temp_model
                    best_seed = temp_seed
        # save the best metrics for each number of states
        if attempts < max_attempts:
            models.append(best_model)
            seeds.append(best_seed)
            LLs.append(best_LL)
            AICs.append(best_model.aic(transformed_df.values, lengths=lengths))
            BICs.append(best_model.bic(transformed_df.values, lengths=lengths))
    model = models[np.argmin(BICs)]
    
    if validation:
        print("Validating...")
        validate_gmmhmm_states(min_states, max_states, LLs, AICs, BICs)
    
    return model.n_components
    


