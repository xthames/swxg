from typing import List, Tuple, Dict 
import numpy as np
import pandas as pd
import datetime as dt
from hmmlearn.hmm import GMMHMM



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
    
    # determine best-fitting number of states for GMMHMM
    return {"transformed_precip": transformed_data}


def gmmhmm_state_num_estimator(transformed_df: pd.DataFrame, min_states: int = 1, max_states: int = 5) -> int:
    """
    Function to programmatically determine the best-fitting number of states 
    for the Gaussian mixture model hidden Markov model
    
    Parameters
    ----------
    transformed_df: pd.DataFrame
        DataFrame for the log10-transformed annualized precipitation values, with
        associated sites and years
    min_states: int, optional
        The minimum number of hidden states to try fitting. Default: 1 
    max_states: int, optional
        The maximum number of hidden states to try fitting. More than ~6 tends
        to perform poorly in terms of best fit and length of computation. Default: 5 
    
    Returns
    -------
    best_num_states: int
        Best-fitting model's number of states, as determined by Bayesian Information
        Criterion (BIC). Log-likelihood and Akaike Information Criterion (AIC) are 
        also used, but log-likelihood is monotonically increasing with number of states
        and AIC does not penalize additional states strongly enough. 
    """

    # separate data from load in
    transformedValues, seqLengths = annualDict["df"].values, annualDict["lengths"]
    nStations = len(annualDict["df"].columns)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # fit a multidimensional Gaussian mixture HMM to data
    # -- note: the covergence EM method can nortoriously can fall into local optima
    # -- suggested fix: run a few fits, take the one with the highest log-likelihood
    # -- when checking states: use AIC and BIC criteria
    models, seeds, AICs, BICs, LLs = [], [], [], [], []
    for numStates in range(minStates, maxStates + 1):
        bestModel, bestLL, bestSeed = None, None, None
        for _ in range(10):
            # define our hidden Markov model, whose states' covariance must be positive definite
            positiveDefinite, tempSeed, tempModel = False, None, None
            while not positiveDefinite:
                # get the random state
                tempSeed = random.getstate()

                # define the parameters for the model
                tempModelObject = GMMHMM(n_components=numStates, n_iter=1000, covariance_type="full", init_params="cmw")
                tempModelObject.startprob_ = np.full(shape=numStates, fill_value=1./numStates)
                tempModelObject.transmat_ = np.full(shape=(numStates, numStates), fill_value=1./numStates)
                tempModel = tempModelObject.fit(transformedValues, lengths=seqLengths)

                # get, reshape the covariance matrices
                tempCovars = tempModel.covars_.reshape((numStates, nStations, nStations))

                # check that each state has a covariance matrix that is positive definite, meaning:
                # -- it's symmetric
                symmetricCheck = all([np.allclose(tempCovars[i].T, tempCovars[i]) for i in range(numStates)])
                # -- it's eigenvalue are all positive
                eigCheck = all([(np.linalg.eigvalsh(tempCovars[i]) > 0).all() for i in range(numStates)])

                # state if the covariance matrix is positive definite or not
                positiveDefinite = symmetricCheck and eigCheck

            # calculate the loglikelihood of this model
            tempScore = tempModel.score(transformedValues, lengths=seqLengths)
            # get the model with the highest score, set that as the best one for each test of states
            if not bestLL or tempScore > bestLL:
                bestLL = tempScore
                bestModel = tempModel
                bestSeed = tempSeed
        # find the best metrics for each number of states (using each model's BIC)
        models.append(bestModel)
        seeds.append(bestSeed)
        LLs.append(bestLL)
        AICs.append(bestModel.aic(transformedValues, lengths=seqLengths))
        BICs.append(bestModel.bic(transformedValues, lengths=seqLengths))
    # our best model has the lowest BIC
    model = models[np.argmin(BICs)]
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # plot how well each num states did
    ModelCriteriaPlot, axis = plt.subplots()
    axis.grid()
    axis.plot(np.arange(minStates, maxStates + 1), AICs, color="blue", marker="o", label="AIC")
    axis.plot(np.arange(minStates, maxStates + 1), BICs, color="green", marker="o", label="BIC")
    axis2 = axis.twinx()
    axis2.plot(np.arange(minStates, maxStates + 1), LLs, color="orange", marker="o", label="LL")
    axis.legend(handles=axis.lines + axis2.lines)
    axis.set_title("Model Selection Criteria")
    axis.set_xlabel("# States")
    axis.set_ylabel("Criterion Value [-, lower is better]")
    axis2.set_ylabel("Log-Likelihood [-, higher is better]")
    plt.tight_layout()
    ModelCriteriaPlot.savefig(plotsDir + r"/gmmhmm/{}/{}_ModelSelectionCriteria.svg".format(dataRepo, repoName))
    plt.close()

    # return the best-fitting number of components
    return model.n_components

