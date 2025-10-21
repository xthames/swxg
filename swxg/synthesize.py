import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats
import statsmodels
import warnings

from .make_figures import *


def synthesize_data(n: int,
                    data: pd.DataFrame,
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
    n: int
        Number of years for the generator to synthesize weather for
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
    
    # synthesize kwargs    
    default_synthesize_kwargs = {"validation_samplesize_mult": 10,
                                 "figure_extension": "svg"}
    if not synthesize_kwargs: 
        synthesize_kwargs = default_synthesize_kwargs
    else:
        for k in default_synthesize_kwargs:
            if k not in synthesize_kwargs:
                synthesize_kwargs[k] = default_synthesize_kwargs[k]
    
    # validation
    global do_validation, validation_dirpath, validation_extension
    do_validation, validation_dirpath, validation_extension = validate, dirpath, synthesize_kwargs["figure_extension"]
     
    # remove all the rows with partially full or unfit years
    filtered_data = data[data["YEAR"].isin(precip_dict["log10_annual_precip"].index.values)]
    incomplete_years, full_years = [], []
    for year in sorted(set(filtered_data["YEAR"].values)):
        if len(set(filtered_data.loc[filtered_data["YEAR"] == year, "MONTH"].values)) == 12:
            full_years.append(int(year))
        else:
            incomplete_years.append(int(year))
    filtered_data = filtered_data[filtered_data["YEAR"].isin(full_years)]
    filtered_data.reset_index(drop=True, inplace=True)
    if resolution == "daily" and "DAY" in filtered_data.columns:
        filtered_data.astype({"SITE": str, "YEAR": int, "MONTH": int, "DAY": int, "PRECIP": float, "TEMP": float})
    else:
        filtered_data.astype({"SITE": str, "YEAR": int, "MONTH": int, "PRECIP": float, "TEMP": float})

    # collecting kNN years
    global kNN_dict
    kNN_dict = {"precip": [], "temp": {}}

    # synthesizing precipitation
    synth_precip = synthesize_precip(n, filtered_data, precip_dict, resolution, incomplete_years) 
    
    # conditionally synthesizing temperature from precipitation
    synth_pt = synthesize_pt_pairs(synth_precip, copulaetemp_dict, filtered_data, resolution) 

    # compare synth data to obs (if requested)
    if do_validation:
        print("Validating generated weather, this may take a moment...")
        add_samples = (synthesize_kwargs["validation_samplesize_mult"] - 1) * n
        if add_samples == 0:
            compare_synth_to_obs(validation_dirpath, validation_extension, synth_pt, filtered_data)
        else:
            kNN_dict = {"precip": [], "temp": {}}
            synth_precip_compare = synthesize_precip(add_samples, filtered_data, precip_dict, resolution, incomplete_years)
            synth_pt_compare = synthesize_pt_pairs(synth_precip_compare, copulaetemp_dict, filtered_data, resolution)
            synth_pt_compare["YEAR"] = synth_pt_compare["YEAR"].values + max(synth_pt["YEAR"].values)
            compare_pt = pd.concat([synth_pt, synth_pt_compare])
            if resolution == "daily" and "DAY" in compare_pt.columns:
                compare_pt.sort_values(by=["SITE", "YEAR", "MONTH", "DAY"], inplace=True)
            else:
                compare_pt.sort_values(by=["SITE", "YEAR", "MONTH"], inplace=True)
            compare_pt.reset_index(drop=True, inplace=True)
            compare_synth_to_obs(validation_dirpath, validation_extension, compare_pt, filtered_data)

    return synth_pt


def synthesize_precip(n_synth_years: int, data: pd.DataFrame, p_dict: dict, resolution: str, incomp_years: list[int]) -> np.array:
    """
    Manager function to synthesize precipitation

    Parameters
    ----------
    n_synth_years: int
        Number of years for the generator to synthesize weather for 
    data: pd.DataFrame
        Observed precipitation and temperature data
    p_dict: dict
        GMMHMM best-fitted model and corresponding parameters
    resolution: str
        Resolution of desired synthesized data
    incomp_years: list[int]
        List of years without the full set of months

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            year_obs_pair = np.reshape([[years[i], obs_spatial_avg[i]] for i in range(len(years))], newshape=(len(years), 2))
        kNN_selected_years = np.full(shape=precip_log10annual_sample.shape[0], fill_value=np.nan)
        disaggregated_sample = np.full(shape=(precip_log10annual_sample.shape[0], n_months, len(sites)), fill_value=np.nan)
        for j, sum_synth_year in enumerate(synth_spatial_avg):
            # (4) calculate Manhattan distance (since 1D) between individual synthetic and all obs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                year_synth_dist = np.reshape([[year_obs_pair[i, 0], abs(sum_synth_year - year_obs_pair[i, 1])] for i in range(len(years))], newshape=(len(years), 2))
            sorted_year_dist = year_synth_dist[year_synth_dist[:, 1].argsort()]
            # (5) choose a year from the set using pre-determined weights
            kNN_selected_years[j] = rng.choice(sorted_year_dist[:k, 0], p=w)
        kNN_dict["precip"] = kNN_selected_years

        # (6) maintain temporal proportionality: synth_{month} = synth_{year} * hist_{month}/hist_{year}
        for i, kNN_selected_year in enumerate(kNN_selected_years):
            year_idx = precip_obs["YEAR"] == kNN_selected_year
            for s, site in enumerate(sites):
                site_idx = precip_obs["SITE"] == site
                if resolution == "monthly" or "DAY" not in precip_obs.columns:
                    kNN_selected_monthlies = precip_obs.loc[year_idx & site_idx, "PRECIP"].values
                else:
                    kNN_selected_monthlies = []
                    for m in range(1, n_months+1):
                        month_idx = precip_obs["MONTH"] == m
                        kNN_selected_monthlies.append(np.nansum(precip_obs.loc[month_idx & year_idx & site_idx, "PRECIP"].values))
                    kNN_selected_monthlies = np.array(kNN_selected_monthlies)
                disaggregated_sample[i, :, s] = synth_annual[i, s] * (kNN_selected_monthlies / sum(kNN_selected_monthlies))

        return disaggregated_sample
 
    rng = np.random.default_rng()
    filtered_annual_data = p_dict["log10_annual_precip"].drop(incomp_years)
    sites, years = sorted(set(data["SITE"].values)), sorted(set(filtered_annual_data.index.values))
    annual_sample = p_dict["model"].sample(n_synth_years)[0]
    precip_data = data[data.columns[:list(data.columns).index("PRECIP")+1]].copy()
    return precip_kNN_disaggregation(annual_sample, precip_data, filtered_annual_data) 


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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
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

    def temp_kNN_disaggregation(t_monthly_synth: np.array, pt_monthly_obs: pd.DataFrame, mnth: int) -> np.array:
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            year_obs_pair = np.reshape([[years[i], obs_spatial_avg[i]] for i in range(len(years))], newshape=(len(years), 2))
        kNN_selected_years = np.full(shape=len(synth_spatial_avg), fill_value=np.nan)
        for j, sa_synth_year in enumerate(synth_spatial_avg):
            # (3) calculate Manhattan distance (since 1D) between individual synthetic and all obs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                year_synth_dist = np.reshape([[year_obs_pair[i, 0], abs(sa_synth_year - year_obs_pair[i, 1])] for i in range(len(years))], newshape=(len(years), 2))
            sorted_year_dist = year_synth_dist[year_synth_dist[:, 1].argsort()]
            # (4) choose which year from the set of years using pre-determined weights
            kNN_selected_years[j] = rng.choice(sorted_year_dist[:k, 0], p=w)
        if mnth not in kNN_dict:
            kNN_dict["temp"][mnth] = kNN_selected_years

        # (5) construct a vector of spatial averages that match the selected kNN years, 
        # -- add noise to smooth out the emergent non-parametric banding
        kNN_spatial_avg = np.full(shape=len(synth_spatial_avg), fill_value=np.nan)
        for i, kNN_selected_year in enumerate(kNN_selected_years):
            knn_year_idx = pt_monthly_obs["YEAR"] == kNN_selected_year
            kNN_selected_station_values = pt_monthly_obs.loc[knn_year_idx & obs_month_idx, "TEMP"].values
            kNN_spatial_avg[i] = np.nanmean(kNN_selected_station_values)
        resids = synth_spatial_avg - kNN_spatial_avg
        disaggregated_sample = np.full(shape=(len(synth_spatial_avg), len(sites)), fill_value=np.nan)
        for i, kNN_selected_year in enumerate(kNN_selected_years):
            knn_year_idx = pt_monthly_obs["YEAR"] == kNN_selected_year
            kNN_selected_station_values = pt_monthly_obs.loc[knn_year_idx & obs_month_idx, "TEMP"].values 
            noise = rng.normal(loc=0., scale=np.nanstd(resids), size=1) 
            disaggregated_sample[i, :] = ((synth_spatial_avg[i] + 273.15) * 
                                          ((kNN_selected_station_values + 273.15) / (np.nanmean(kNN_selected_station_values) + 273.15 + noise))) - 273.15 

        return disaggregated_sample
    
    def daily_kNN_disaggregation(synth_month_df: pd.DataFrame, obs_daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Using previously discovered k-NN years and months, convert monthly
        WX data to daily data. This technique is non-parametric and therefore 
        requires existing observations
        
        Parameters
        ----------
        synth_month_df: pd.DataFrame
            Synthetic precipitation and temperature at monthly resolution
        obs_daily_df: pd.DataFrame
            Observed precipitation and temperature at daily resolution

        Returns
        -------
        disaggregated_sample: pd.DataFrame
            k-NN disaggregated daily synthetic data
        """
        
        # average precip values per doy from observations
        doy_dict = {}
        for site in sorted(set(obs_daily_df["SITE"].values)):
            doy_site_idx = obs_daily_df["SITE"] == site
            doy_site_entry = obs_daily_df.loc[doy_site_idx]
            if site not in doy_dict: doy_dict[site] = {}
            for i in range(doy_site_entry.shape[0]):
                row_entry = doy_site_entry.iloc[i]
                doy = int(dt.datetime.strptime("{}-{}-{}".format(int(row_entry["YEAR"]), str(row_entry["MONTH"]).zfill(2), str(row_entry["DAY"]).zfill(2)), "%Y-%m-%d").strftime("%j"))
                if doy not in doy_dict[site]:
                    doy_dict[site][doy] = [row_entry["PRECIP"]]
                else:
                    doy_dict[site][doy].append(row_entry["PRECIP"])
            for doy in doy_dict[site]:
                doy_dict[site][doy] = float(np.nanmean(doy_dict[site][doy]))

        daily_dict = {}
        for site in sites:
            monthly_site_idx = synth_month_df["SITE"] == site
            monthly_site_entry = synth_month_df.loc[monthly_site_idx] 
            daily_site_idx = obs_daily_df["SITE"] == site
            daily_site_entry = obs_daily_df.loc[daily_site_idx]
            for mnth in sorted(set(monthly_site_entry["MONTH"].values)):    
                monthly_month_idx = monthly_site_entry["MONTH"] == mnth
                monthly_month_entry = monthly_site_entry.loc[monthly_month_idx]
                daily_month_idx = daily_site_entry["MONTH"] == mnth
                daily_month_entry = daily_site_entry.loc[daily_month_idx]
                for y in range(len(set(synth_month_df["YEAR"].values))):
                    monthly_year_idx = monthly_month_entry["YEAR"] == y+1
                    monthly_year_entry = monthly_month_entry.loc[monthly_year_idx]
                    daily_precip_year_idx = daily_month_entry["YEAR"] == kNN_dict["precip"][y]
                    daily_precip_year_entry = daily_month_entry.loc[daily_precip_year_idx]
                    daily_temp_year_idx = daily_month_entry["YEAR"] == kNN_dict["temp"][mnth][y]
                    daily_temp_year_entry = daily_month_entry.loc[daily_temp_year_idx]
                    sync_num_days = min([daily_precip_year_entry.shape[0], daily_temp_year_entry.shape[0]]) 
                    if np.nansum(daily_precip_year_entry["PRECIP"].values[:sync_num_days]) == 0:
                        sync_doys = []
                        for d in range(sync_num_days):
                            sync_doys.append(int(dt.datetime.strptime("{}-{}-{}".format(int(kNN_dict["precip"][y]),str(mnth).zfill(2),str(d+1).zfill(2)), "%Y-%m-%d").strftime("%j")))
                        sync_precips = [doy_dict[site][sync_doy] for sync_doy in sync_doys]
                    else:
                        sync_precips = daily_precip_year_entry["PRECIP"].values[:sync_num_days]
                    sync_temps = daily_temp_year_entry["TEMP"].values[:sync_num_days]
                    for day in range(sync_num_days): 
                        prcp = sync_precips[day] * (monthly_year_entry["PRECIP"].values[0] / np.nansum(sync_precips))
                        temp = sync_temps[day] + (monthly_year_entry["TEMP"].values[0] - np.nanmean(sync_temps))
                        daily_dict[(site, y+1, mnth, day+1)] = [site, y+1, mnth, day+1, round(prcp, 4), round(temp, 2)] 
        daily_df = pd.DataFrame().from_dict(daily_dict, orient="index", columns=["SITE", "YEAR", "MONTH", "DAY", "PRECIP", "TEMP"])
        daily_df.sort_values(by=["SITE", "YEAR", "MONTH", "DAY"], inplace=True)
        daily_df.reset_index(drop=True, inplace=True)
        daily_df.astype({"SITE": str, "YEAR": int, "MONTH": int, "DAY": int, "PRECIP": float, "TEMP": float}) 
        return daily_df

    rng = np.random.default_rng()
    sites = sorted(set(pt_df["SITE"].values))
    years = [y for y in range(min(pt_df["YEAR"].values), max(pt_df["YEAR"].values)+1)]
    month_names, month_vals = list(t_dict.keys()), [m+1 for m in range(len(t_dict.keys()))]
    n_synth_years, n_months, n_sites = synth_prcp.shape
    synth_monthly_df = pd.DataFrame(columns=["SITE", "YEAR", "MONTH", "PRECIP", "TEMP"])
    synth_monthly_df["SITE"] = np.repeat(sites, n_synth_years * n_months)
    synth_monthly_df["YEAR"] = list(np.repeat([y+1 for y in range(n_synth_years)], n_months)) * n_sites
    synth_monthly_df["MONTH"] = month_vals * (n_synth_years * n_sites)

    if resolution == "daily" and "DAY" in pt_df.columns:
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
        pt_monthly_df = pt_df
    
    for m, month in enumerate(month_names):
        month_idx = synth_monthly_df["MONTH"] == month_vals[m]
        
        # spatially average synth precip for the month
        sa_synth_prcp = synth_prcp[:, m, :].mean(axis=1)
        
        # transform the synthetic preciptation to residuals using the ARfit used in the copulas
        nP = len(sa_synth_prcp)
        ar1_synth_prcp_fit = t_dict[month]["PRECIP ARFit"].apply(sa_synth_prcp)
        full_ar1_prcp = np.array([np.nanmean(ar1_synth_prcp_fit.fittedvalues), *ar1_synth_prcp_fit.fittedvalues])
        resid_prcp = sa_synth_prcp - full_ar1_prcp

        # transform into uniform marginals
        uP = stats.rankdata(resid_prcp, method="average") / (nP+1)

        # conditional simulation of the uT | uP --> coming from {d/d(uP) [C(uP, uT)]}^{-1}
        uT = conditionally_simulate_uT(uP, t_dict[month]["BestCopula"])

        # transform from marginals to residuals (using CDF^{-1}) to data (using AR fit params)
        resid_temp = t_dict[month]["TEMP Resid Dist"].ppf(uT)
        obs_mean, obs_std = np.nanmean(t_dict[month]["TEMP"].astype(float)), np.nanstd(t_dict[month]["TEMP"].astype(float))
        sa_synth_temp_approx = (resid_temp + t_dict[month]["TEMP ARFit"].params[0]) / (1. - t_dict[month]["TEMP ARFit"].params[1]) 
        approx_mean, approx_std = np.nanmean(sa_synth_temp_approx), np.nanstd(sa_synth_temp_approx)
        sa_synth_temp = (sa_synth_temp_approx - approx_mean)*(obs_std/approx_std) + obs_mean

        # (parametric) conditional temperature can sample values WAY too high or low
        # -- if this happens, resample the conditional temperatures until it doesn't happen
        obs_temp = t_dict[month]["TEMP"].astype(float)
        obs_max_diff = np.abs(np.nanmax(obs_temp) - np.nanmin(obs_temp))
        while np.any(sa_synth_temp < np.nanmin(obs_temp) - obs_max_diff) or np.any(sa_synth_temp > np.nanmax(obs_temp) + obs_max_diff):
            uT = conditionally_simulate_uT(uP, t_dict[month]["BestCopula"])
            resid_temp = t_dict[month]["TEMP Resid Dist"].ppf(uT)
            sa_synth_temp_approx = (resid_temp + t_dict[month]["TEMP ARFit"].params[0]) / (1. - t_dict[month]["TEMP ARFit"].params[1]) 
            approx_mean, approx_std = np.nanmean(sa_synth_temp_approx), np.nanstd(sa_synth_temp_approx)
            sa_synth_temp = (sa_synth_temp_approx - approx_mean)*(obs_std/approx_std) + obs_mean

        # take the spatially-averaged temperatures and disaggregate to return per-station values
        synth_temp = temp_kNN_disaggregation(sa_synth_temp, pt_monthly_df, month_vals[m])
        
        # assign to dataframe 
        for s, site in enumerate(sites):
            site_idx = synth_monthly_df["SITE"] == site
            synth_monthly_df.loc[month_idx & site_idx, "PRECIP"] = synth_prcp[:, m, s]            
            synth_monthly_df.loc[month_idx & site_idx, "TEMP"] = synth_temp[:, s] 
    
    if resolution == "monthly" or (resolution == "daily" and "DAY" not in pt_df.columns):
        if resolution == "daily" and "DAY" not in pt_df.columns:
            warnings.warn("Input dataset at monthly resolution cannot be disaggregated to daily! Returning monthly...", UserWarning)
        return synth_monthly_df
    else:
        return daily_kNN_disaggregation(synth_monthly_df, pt_df) 

