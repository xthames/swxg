import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats
import copy

from .make_figures import *


def synthesize_data(data: pd.DataFrame,
                    precip_dict: dict,
                    copulaetemp_dict: dict,
                    resolution: str,
                    validate: bool,
                    dirpath: str,
                    synthesize_kwargs: dict) -> pd.DataFrame:
    """
    Managing function that synthesizes weather from the fit or given 
    statistical parameters 

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe of formatted weather data to reference
    precip_dict: dict
        Dictionary of fit precipitation parameters 
    copulaetemp_dict: dict
        Dictionary of fit copula parameters to conditionally construct temperature 
    resolution: str
        The temporal resolution of the  data 
    validate: bool
        Flag for producing figures to validate each step of the generator
    dirpath: str
        Path for where to save the validation figures
    synthesize_kwargs: dict
        Keyword arguments related to the fit
    
    Returns
    -------
    synth_data: pd.DataFrame
        The synthesized weather data from the given observations and parameters
    """
    
    # validation
    global do_validation, validation_dirpath
    do_validation, validation_dirpath = validate, dirpath

    # synthesize kwargs    
    default_synthesize_kwargs = {}
    if not synthesize_kwargs: 
        synthesize_kwargs = default_synthesize_kwargs
    else:
        for k in default_synthesize_kwargs:
            if k not in synthesize_kwargs:
                synthesize_kwargs[k] = default_synthesize_kwargs[k]
     
    # synthesizing precipitation
    synth_precip = synthesize_precip(data, precip_dict, resolution) 
    
    # conditionally synthesizing temperature from precipitation
    synth_pt = synthesize_pt_pairs(synth_precip, copulaetemp_dict, data, resolution) 
    
    return synth_pt


def synthesize_precip(data: pd.DataFrame, p_dict: dict, resolution: str) -> np.array:
    """
    Manager function to synthesize precipitation

    Parameters
    ----------
    data: pd.DataFrame
        Observed precipitation and temperature data
    p_dict: dict
        GMMHMM best-fitted model and corresponding parameters
    resolution: str
        Time resolution of the synthesized data

    Returns
    -------
    synth_precip_monthly: pd.DataFrame
        Synthesized monthly precipitation
    """
    
    def precip_kNN_disaggregation(precip_log10annual_sample: np.array, 
                                  precip_obs: pd.DataFrame, 
                                  precip_log10annual_obs: pd.DataFrame) -> np.array:
        """
        Function to perform a k-NN disaggregation scheme for the log10(annual) precipitation 
        data specifically, adapted from ideas in Lall & Sharma (1996), 
        Apipattanavis et al. (2007), Nowak et al. (2010), and Quinn et al. (2020, supplmental). 
        This technique is non-parametric and therefore requires existing observations
        
        Parameters
        ----------
        precip_log10annual_sample: np.array
            Sample taken from the GMMHMM of the log10 annual precipitation data
        precip_obs: pd.DataFrame
            Observed precipitation data at the appropriate resolution
        precip_log10annual_obs: pd.DataFrame
            log10(annual)-transformed precipitation data as determined in
            the GMMHMM fit

        Returns
        -------
        disaggregated_sample: pd.DataFrame
            k-NN disaggregated monthly precipitation sample
        """

        # (0) convert to real-space from synth log-space
        n_months = 12
        synth_annual = 10.**precip_log10annual_sample

        # (1) create k, weights -- k recommended to be int(sqrt(n)), where n is the number of years
        k = round(np.sqrt(len(years)))
        w = np.array([(1 / j) for j in range(1, k+1)]) / sum([(1 / j) for j in range(1, k+1)])

        # (2) spatial averages for observations and sample
        obs_spatial_avg = np.nanmean(10.**precip_log10annual_obs.values, axis=1)
        synth_spatial_avg = np.nanmean(synth_annual, axis=1)

        # (3) choose one of the k closest observed years 
        year_obs_pair = np.reshape([[years[i], obs_spatial_avg[i]] for i in range(len(years))], newshape=(len(years), 2))
        kNN_selected_years = np.full(shape=len(years), fill_value=np.NaN)
        disaggregated_sample = np.full(shape=(len(years), n_months, len(sites)), fill_value=np.NaN)
        for j, sum_synth_year in enumerate(synth_spatial_avg):
            # (4) calculate Manhattan distance (since 1D) between individual synthetic and all obs
            year_synth_dist = np.reshape([[year_obs_pair[i, 0], abs(sum_synth_year - year_obs_pair[i, 1])] for i in range(len(years))], newshape=(len(years), 2))
            sorted_year_dist = year_synth_dist[year_synth_dist[:, 1].argsort()]
            # (5) choose a year from the set using pre-determined weights
            kNN_selected_years[j] = rng.choice(sorted_year_dist[:k, 0], p=w)

        # (6) maintain temporal proportionality: synth_{month}/synth_{year} = synth_{year} * hist_{month}/hist_{year}
        for i, year in enumerate(years):
            year_idx = precip_obs["YEAR"] == kNN_selected_years[i]
            for s, site in enumerate(sites):
                site_idx = precip_obs["SITE"] == site
                if resolution == "monthly":
                    kNN_selected_monthlies = precip_obs.loc[year_idx & site_idx, "PRECIP"].values
                else:
                    kNN_selected_monthlies = []
                    for m in range(1, n_months+1):
                        month_idx = precip_obs["MONTH"] == m
                        kNN_selected_monthlies.append(np.nansum(precip_obs.loc[month_idx, "PRECIP"].values))
                    kNN_selected_monthlies = np.array(kNN_selected_monthlies)
                disaggregated_sample[i, :, s] = synth_annual[i, s] * (kNN_selected_monthlies / sum(kNN_selected_monthlies))

        return disaggregated_sample
 
    rng = np.random.default_rng()
    sites, years = sorted(set(data["SITE"].values)), [y for y in range(min(data["YEAR"].values), max(data["YEAR"].values)+1)]
    annual_sample = p_dict["model"].sample(len(years))[0]
    precip_data = data[data.columns[:list(data.columns).index("PRECIP")+1]].copy()
    annual_data = p_dict["log10_annual_precip"]
    return precip_kNN_disaggregation(annual_sample, precip_data, annual_data) 


def synthesize_pt_pairs(synth_prcp: np.array, t_dict: dict, pt_df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """
    Manager function to conditionally synthesize temperature from 
    precipitation

    Parameters
    ----------
    synth_prcp: pd.DataFrame
        Synthesized precipitation at monthly resolution
    t_dict: dict
        Copula best-fitted, spatially-averaged models and corresponding parameters
    pt_df: pd.DataFrame
        Temporally-formatted observed precipitation and temperature data
    resolution: str
        Time resolution of the synthesized data

    Returns
    -------
    synth_monthly_df: pd.DataFrame
        Synthesized monthly precipitation and temperature
    """
    
    def conditionally_simulate_uT(poP: np.array, cop_list: list) -> np.array:
        """
        Conditionally simulate temperature pseudo-observations from
        synthesized precipitation pseudo-observations using the 
        appropriate copula family

        Parameters
        ----------
        poP: np.array
            Synthesized precipitation pseudo-observations
        cop_list: list
            List with the copula object and family name

        Returns
        -------
        poT: np.array
            Conditionally sampled temperature pseudo-observations
        """

        cop_obj, cop_name = cop_list[0], cop_list[1]
        if cop_name == "Independence":
            # v = d/du [C(u,v)] --> since C(u,v) = u*v, marginal v *is* the inverse of the conditional CDF
            poT = rng.random(size=len(poP))
        if cop_name == "Frank":
            # v = inverse of the conditional CDF -- c(v|u)^{-1} -- so the ppf of the copula given u
            y = rng.random(size=len(poP))
            try:
                poT = cop_obj.percent_point(y, poP)
            except ValueError:
                y = rng.random(size=len(poP))
                poT = cop_obj.percent_point(y, poP)
        if cop_name == "Gaussian":
            # (1) conditional sampling starts with the Cholesky decomposition of the Gaussian parameter
            # (2) transform to normal distribution for poP, generate on normal distribution for y
            # (3) matrix multiply to get sample simulation given input
            # (4) temperature marginals are the cdf of the temperature half of the conditional sample
            A = np.tril(np.linalg.cholesky(cop_obj.correlation.values))
            normP = scipy.stats.norm.ppf(poP)
            y = scipy.stats.norm.ppf(rng.random(size=len(poP)))
            cond_samp = A @ np.array([normP, y])
            poT = scipy.stats.norm.cdf(cond_samp[1])

        return poT

    def temp_kNN_disaggregation(t_monthly_synth: np.array, pt_monthly_obs: pd.DataFrame, mnth: int):
        """
        Function to perform a k-NN disaggregation scheme for the spatially-averaged
        temperature data specifically, adapted from ideas in Lall & Sharma (1996), 
        Apipattanavis et al. (2007), Nowak et al. (2010), and Quinn et al. (2020, supplmental). 
        This technique is non-parametric and therefore requires existing observations
        
        Parameters
        ----------
        t_monthly_synth: np.array
            Synthesized spatially-averaged temperature at monthly resolution
        pt_monthly_obs: pd.DataFrame
            Observed precipitation and temperature at monthly resolution
        mnth: int
            Month of the year, as an integer value

        Returns
        -------
        disaggregated_sample: pd.DataFrame
            k-NN disaggregated monthly temperature sample
        """
        
        k = round(np.sqrt(len(years)))
        w = np.array([(1 / j) for j in range(1, k+1)]) / sum([(1 / j) for j in range(1, k+1)])
 
        # (1) spatial averages for obs and synth
        obs_month_idx = pt_monthly_obs["MONTH"] == mnth
        obs_spatial_avg = []
        for year in years:
            obs_year_idx = pt_monthly_obs["YEAR"] == year
            obs_temps = pt_monthly_obs.loc[obs_month_idx & obs_year_idx, "TEMP"].values
            obs_spatial_avg.append(np.inf if all(np.isnan(obs_temps)) else np.nanmean(obs_temps)) 
        obs_spatial_avg = np.array(obs_spatial_avg)
        synth_spatial_avg = t_monthly_synth 

        # (2) link the summed historic year with the year itself, empty vector to fill with year choices
        year_obs_pair = np.reshape([[years[i], obs_spatial_avg[i]] for i in range(len(years))], newshape=(len(years), 2))
        kNN_selected_years = np.full(shape=len(years), fill_value=np.NaN)
        for j, sa_synth_year in enumerate(synth_spatial_avg):
            # (3) calculate Manhattan distance (since 1D) between individual synthetic and all obs
            year_synth_dist = np.reshape([[year_obs_pair[i, 0], abs(sa_synth_year - year_obs_pair[i, 1])] for i in range(len(years))], newshape=(len(years), 2))
            sorted_year_dist = year_synth_dist[year_synth_dist[:, 1].argsort()]
            # (4) choose which year from the set of years using pre-determined weights
            kNN_selected_years[j] = rng.choice(sorted_year_dist[:k, 0], p=w)
        
        # (5) construct a vector of spatial averages that match the selected kNN years, 
        # -- add noise to smooth out the emergent non-parametric banding
        kNN_spatial_avg = np.full(shape=(len(years)), fill_value=np.NaN)
        for i in range(len(years)):
            knn_year_idx = pt_monthly_obs["YEAR"] == kNN_selected_years[i]
            kNN_selected_station_values = pt_monthly_obs.loc[knn_year_idx & obs_month_idx, "TEMP"].values
            kNN_spatial_avg[i] = np.nanmean(kNN_selected_station_values)
        resids = synth_spatial_avg - kNN_spatial_avg
        disaggregated_sample = np.full(shape=(len(years), len(sites)), fill_value=np.NaN)
        for i in range(len(years)):
            knn_year_idx = pt_monthly_obs["YEAR"] == kNN_selected_years[i]
            kNN_selected_station_values = pt_monthly_obs.loc[knn_year_idx & obs_month_idx, "TEMP"].values 
            noise = rng.normal(loc=0., scale=np.nanstd(resids), size=1) 
            disaggregated_sample[i, :] = ((synth_spatial_avg[i] + 273.15) * 
                                          ((kNN_selected_station_values + 273.15) / (np.nanmean(kNN_selected_station_values) + 273.15 + noise))) - 273.15 

        return disaggregated_sample
    
    rng = np.random.default_rng()
    sites = sorted(set(pt_df["SITE"].values))
    years = [y for y in range(min(pt_df["YEAR"].values), max(pt_df["YEAR"].values)+1)]
    month_names, month_vals = list(t_dict.keys()), [m+1 for m in range(len(t_dict.keys()))]
    n_years, n_months, n_sites = synth_prcp.shape
    synth_monthly_df = pd.DataFrame(columns=["SITE", "YEAR", "MONTH", "PRECIP", "TEMP"])
    synth_monthly_df["SITE"] = np.repeat(sites, n_years * n_months)
    synth_monthly_df["YEAR"] = list(np.repeat(years, n_months)) * n_sites
    synth_monthly_df["MONTH"] = month_vals * (n_years * n_sites)

    if resolution == "daily":
        pt_monthly_df_dict = {}
        for site in sorted(set(pt_df["SITE"].values)):
            site_idx = pt_df["SITE"] == site
            for year in sorted(set(pt_df["YEAR"].values)):
                year_idx = pt_df["YEAR"] == year
                for month in sorted(set(pt_df["MONTH"].values)):
                    month_idx = pt_df["MONTH"] == month
                    prcps = pt_df.loc[site_idx & year_idx & month_idx, "PRECIP"].values
                    prcp_sum = np.nan if len(prcps) == 0 else np.nansum(prcps)
                    temps = pt_df.loc[site_idx & year_idx & month_idx, "TEMP"].values
                    temp_avg = np.nan if len(temps) == 0 else np.nanmean(temps)
                    pt_monthly_df_dict[(site, year, month)] = [site, year, month, prcp_sum, temp_avg]
        pt_monthly_df = pd.DataFrame().from_dict(pt_monthly_df_dict, orient="index", columns=["SITE", "YEAR", "MONTH", "PRECIP", "TEMP"])
        pt_monthly_df.reset_index(drop=True, inplace=True)
        pt_monthly_df.astype({"SITE": str, "YEAR": int, "MONTH": int, "PRECIP": float, "TEMP": float})
    else:
        pt_monthly_df = pt_obs
    
    for m, month in enumerate(month_names):
        month_idx = synth_monthly_df["MONTH"] == month_vals[m]
        # spatially average synth precip for the month
        sa_synth_prcp = synth_prcp[:, m, :].mean(axis=1)
        
        # transform the synthetic preciptation to residuals using the ARfit used in the copulas
        nP = len(sa_synth_prcp)
        full_ar1_prcp = np.array([np.nanmean(t_dict[month]["PRECIP ARFit"].fittedvalues), *t_dict[month]["PRECIP ARFit"].fittedvalues])
        resid_prcp = sa_synth_prcp - full_ar1_prcp

        # transform into uniform marginals
        uP = stats.rankdata(resid_prcp, method="average") / (nP+1)

        # conditional simulation of the uT | uP --> coming from {d/d(uP) [C(uP, uT)]}^{-1}
        uT = conditionally_simulate_uT(uP, t_dict[month]["BestCopula"])

        # transform from marginals to residuals (using CDF^{-1}) to data (using AR fit)
        resid_temp = t_dict[month]["TEMP Resid Dist"].ppf(uT)
        full_ar1_temp = np.array([np.nanmean(t_dict[month]["TEMP ARFit"].fittedvalues), *t_dict[month]["TEMP ARFit"].fittedvalues])
        sa_synth_temp = resid_temp + full_ar1_temp

        # (parametric) conditional temperature can sample values WAY too high or low
        # -- if this happens, resample the conditional temperatures until it doesn't happen
        obs_temp = t_dict[month]["TEMP"].astype(float)
        obs_max_diff = np.abs(np.nanmax(obs_temp) - np.nanmin(obs_temp))
        while np.any(sa_synth_temp < np.nanmin(obs_temp) - obs_max_diff) or np.any(sa_synth_temp > np.nanmax(obs_temp) + obs_max_diff):
            uT = conditionally_simulate_uT(uP, t_dict[month]["BestCopula"])
            resid_temp = t_dict[month]["TEMP Resid Dist"].ppf(uT)
            sa_synth_temp = resid_temp + full_ar1_temp

        # take the spatially-averaged temperatures and disaggregate to return per-station values
        synth_temp = temp_kNN_disaggregation(sa_synth_temp, pt_monthly_df, month_vals[m])
        
        # assign to dataframe 
        for s, site in enumerate(sites):
            site_idx = synth_monthly_df["SITE"] == site
            synth_monthly_df.loc[month_idx & site_idx, "PRECIP"] = synth_prcp[:, m, s]            
            synth_monthly_df.loc[month_idx & site_idx, "TEMP"] = synth_temp[:, s] 
    
    if resolution == "monthly":
        synth_df = synth_monthly_df
    else:
        synth_df_dict = {}
        for site in sorted(set(synth_monthly_df["SITE"].values)):
            obs_site_idx = pt_df["SITE"] == site
            synth_site_idx = synth_monthly_df["SITE"] == site
            for year in sorted(set(synth_monthly_df["YEAR"].values)):
                obs_year_idx = pt_df["YEAR"] == year
                synth_year_idx = synth_monthly_df["YEAR"] == year
                for month in sorted(set(synth_monthly_df["MONTH"].values)):
                    obs_month_idx = pt_df["MONTH"] == month
                    obs_monthly_entry = pt_df.loc[obs_site_idx & obs_year_idx & obs_month_idx]
                    synth_month_idx = synth_monthly_df["MONTH"] == month
                    synth_monthly_entry = synth_monthly_df.loc[synth_site_idx & synth_year_idx & synth_month_idx]
                    obs_daily_prcps, obs_daily_temps = obs_monthly_entry["PRECIP"].values, obs_monthly_entry["TEMP"].values
                    obs_prcp_sum = np.nan if len(obs_daily_prcps) == 0 or all(np.isnan(obs_daily_prcps)) else np.nansum(obs_daily_prcps)
                    obs_temp_avg = np.nan if len(obs_daily_temps) == 0 or all(np.isnan(obs_daily_temps)) else np.nanmean(obs_daily_temps)
                    for day, pT in enumerate(zip(obs_daily_prcps, obs_daily_temps)):
                        obs_daily_prcp = synth_monthly_entry["PRECIP"].values[0] * (pT[0] / obs_prcp_sum)
                        obs_daily_temp = synth_monthly_entry["TEMP"].values[0] + (pT[1] - obs_temp_avg)
                        synth_df_dict[(site, year, month, day+1)] = [site, year, month, day+1, obs_daily_prcp, obs_daily_temp]
        synth_df = pd.DataFrame().from_dict(synth_df_dict, orient="index", columns=["SITE", "YEAR", "MONTH", "DAY", "PRECIP", "TEMP"])
        synth_df.reset_index(drop=True, inplace=True)
        synth_df.astype({"SITE": str, "YEAR": int, "MONTH": int, "DAY": int, "PRECIP": float, "TEMP": float})

    return synth_df 
