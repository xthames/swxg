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
    synth_pt = synthesize_pt_pairs(synth_precip, precip_dict, copulaetemp_dict, resolution) 

    return pd.DataFrame()


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


def synthesize_pt_pairs(synth_prcp, p_dict, t_dict, resolution) -> pd.DataFrame:
    """
    Manager function to conditionally synthesize temperature from 
    precipitation

    Parameters
    ----------
    synth_prcp: pd.DataFrame
        Synthesized precipitation at monthly resolution
    p_dict: dict
        GMMHMM best-fitted model and corresponding parameters
    t_dict: dict
        Copula best-fitted, spatially-averaged models and corresponding parameters
    resolution: str
        Time resolution of the synthesized data

    Returns
    -------
    synth_precip_monthly: pd.DataFrame
        Synthesized monthly precipitation
    """
    
    # conditional simulation of CDF(uT | uP)
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

    def temp_kNN_disaggregation(pt_monthly_synth: np.array, pt_obs: pd.DataFrame, month: int):
        """
        Function to perform a k-NN disaggregation scheme for the spatially-averaged
        temperature data specifically, adapted from ideas in Lall & Sharma (1996), 
        Apipattanavis et al. (2007), Nowak et al. (2010), and Quinn et al. (2020, supplmental). 
        This technique is non-parametric and therefore requires existing observations
        
        Parameters
        ----------
        pt_monthly_synth: np.array
            Synthesized preciptiation and temperature at monthly resolution
        pt_obs: pd.DataFrame
            Observed precipitation and temperature
        month: int
            Month of the year, as an integer value

        Returns
        -------
        disaggregated_sample: pd.DataFrame
            k-NN disaggregated monthly temperature sample
        """
        # (0) separating different years
        histYears, completeYears = gmmhmmyears, [y for y in range(min(gmmhmmyears), max(gmmhmmyears)+1)]
        
        # (1) create k, weights
        # -- recommended to be int(sqrt(n)), where n is the number of years in the time-series (Lall & Sharma, 1996)
        k = round(np.sqrt(len(histYears))) if not k else k
        # -- make the weights
        w = np.array([(1 / j) for j in range(1, k+1)]) / sum([(1 / j) for j in range(1, k+1)])

        # (2) spatial averages for historic and synth
        histMonthIdx = historicMonthlyData["MONTH"] == mnth
        histSpatialAvg = []
        for year in histYears:
            histYearIdx = historicMonthlyData["YEAR"] == year
            histSpatialAvg.append(np.nanmean(historicMonthlyData.loc[histMonthIdx & histYearIdx, "TAVG"].values)) 
        histSpatialAvg = np.array(histSpatialAvg)
        synthSpatialAvg = sampleData 

        # (3) link the summed historic year with the year itself, empty vector to fill with year choices
        # noinspection PyTypeChecker
        yearHistPair = np.reshape([[histYears[i], histSpatialAvg[i]] for i in range(len(histYears))], newshape=(len(histYears), 2))
        kNNSelectedYears = np.full(shape=len(completeYears), fill_value=np.NaN)

        # for each year in the synthetic data...
        for j, saSynthYear in enumerate(synthSpatialAvg):
            # (4) calculate Euclidean distance (absolute value for 1D) between individual synthetic and all historical
            # noinspection PyTypeChecker
            yearSynthDist = np.reshape([[yearHistPair[i, 0], abs(saSynthYear - yearHistPair[i, 1])] for i in range(len(histYears))], newshape=(len(histYears), 2))
            # -- ascending sort the years based on distance, only consider first k years
            sortedYearDist = yearSynthDist[yearSynthDist[:, 1].argsort()]
            # (5) choose which year from the set of years using pre-determined weights
            kNNSelectedYears[j] = rng.choice(sortedYearDist[:k, 0], p=w)
        
        # (6) construct a vector of spatial averages that match the selected kNN years, noise to smooth out the non-parametric banding
        kNNSpatialAvg = np.full(shape=(len(completeYears)), fill_value=np.NaN)
        for i in range(len(completeYears)):
            knnYearIdx = historicMonthlyData["YEAR"] == kNNSelectedYears[i]
            kNNSelectedStationValues = historicMonthlyData.loc[knnYearIdx & histMonthIdx, "TAVG"].values
            kNNSpatialAvg[i] = np.nanmean(kNNSelectedStationValues)
        resids = synthSpatialAvg - kNNSpatialAvg

        # -- create 3D array to hold the disaggregated monthly data
        # -- row: years, col: stations
        disaggregatedSample = np.full(shape=(len(completeYears), len(stations)), fill_value=np.NaN)
        # (7) maintain spatial differences relative to shifted mean: 
        # temp_{synth station} = temp_{obs station} + (mean_{stations}(temp_{synth}) - mean_{stations}(temp_{obs}))
        for i in range(len(completeYears)):
            knnYearIdx = historicMonthlyData["YEAR"] == kNNSelectedYears[i]
            kNNSelectedStationValues = historicMonthlyData.loc[knnYearIdx & histMonthIdx, "TAVG"].values 
            #disaggregatedSample[i, :] = kNNSelectedStationValues + (synthSpatialAvg[i] - np.nanmean(kNNSelectedStationValues))
            noise = rng.normal(loc=0., scale=np.nanstd(resids), size=1) 
            disaggregatedSample[i, :] = ((synthSpatialAvg[i] + 273.15) * ((kNNSelectedStationValues + 273.15) / (np.nanmean(kNNSelectedStationValues) + 273.15 + noise))) - 273.15 

        # return the disaggregated sample
        return disaggregatedSample
    
    rng = np.random.default_rng()
    sites = sorted(set(p_dict["log10_annual_precip"].columns.values))
    years = [y for y in range(min(p_dict["log10_annual_precip"].index.values), max(p_dict["log10_annual_precip"].index.values)+1)]
    month_names, month_vals = list(t_dict.keys()), [m+1 for m in range(len(t_dict.keys()))]
    n_years, n_months, n_sites = synth_prcp.shape
    monthly_df = pd.DataFrame(columns=["SITE", "YEAR", "MONTH", "PRECIP", "TEMP"])
    monthly_df["SITE"] = np.repeat(sites, n_years * n_months)
    monthly_df["YEAR"] = list(np.repeat(years, n_months)) * n_sites
    monthly_df["MONTH"] = month_vals * (n_years * n_sites)

    print(synth_prcp)
    print(p_dict)
    print(t_dict)
    print(monthly_df)

    # # for each month in the sample...
    # for m, month in enumerate(months):
    #     monthIdx = synthDF["MONTH"] == month
    #     # the spatially-averaged precipitation data we're interested in
    #     saSynthPRCP = prcpSample[:, m, :].mean(axis=1)
    #     
    #     # transform the synthetic preciptation to residuals using the ARfit used in the copulas
    #     # -- note: first fittedvalues index is NaN, so fill that position with an average value from the rest of the fitted
    #     nP = len(saSynthPRCP)
    #     fullPrecipFitted = np.array([np.nanmean(copulaDict[month]["PRCP ARFit"].fittedvalues), *copulaDict[month]["PRCP ARFit"].fittedvalues])
    #     resid = saSynthPRCP - fullPrecipFitted

    #     # transform into uniform marginals
    #     uP = rankdata(resid, method="average", nan_policy="omit") / (nP+1)

    #     # conditional simulation of the uT | uP --> coming from {d/d(uP) [C(uP, uT)]}^{-1}
    #     # uT = SimulateConditionalUT(uP, copulaDict[station][month]["BestCopula"])
    #     # -- JUST USE FRANK
    #     uT = SimulateConditionalUT(uP, copList=[copulaDict[month]["CopulaDF"].at["Frank", "Copula"], "Frank"])

    #     # transform from marginals to residuals (using CDF^{-1}) to data (using AR fit)
    #     # -- same trick of averaging the fitted values for filling that earliest NaN point
    #     synthTResids = copulaDict[month]["TAVG Resid Dist"].ppf(uT)
    #     fullTavgFitted = np.array([np.nanmean(copulaDict[month]["TAVG ARFit"].fittedvalues), *copulaDict[month]["TAVG ARFit"].fittedvalues])
    #     saSynthTAVG = synthTResids + fullTavgFitted

    #     # sometimes the conditionally simulated temperature values are WAY too high or low
    #     # -- like, hotter than water boiling or colder than absolute zero
    #     # -- if this happens, resample the conditional temperatures until it doesn't happen
    #     histTavg = copulaDict[month]["TAVG"].astype(float)
    #     histTavgDiff = np.abs(np.nanmax(histTavg) - np.nanmin(histTavg))
    #     while np.any(saSynthTAVG < np.nanmin(histTavg) - histTavgDiff) or np.any(saSynthTAVG > np.nanmax(histTavg) + histTavgDiff):
    #         # uT = SimulateConditionalUT(uP, copulaDict[station][month]["BestCopula"])
    #         uT = SimulateConditionalUT(uP, copList=[copulaDict[month]["CopulaDF"].at["Frank", "Copula"], "Frank"])
    #         synthTResids = copulaDict[month]["TAVG Resid Dist"].ppf(uT)
    #         saSynthTAVG = synthTResids + fullTavgFitted

    #     # take the spatially-averaged temperatures and disaggregate to return per-station values
    #     tavgSample = kNNTavgDisaggregation(saSynthTAVG, monthlyDF, month)
    #     
    #     # store precip, temp in the DF 
    #     for s, station in enumerate(stations):
    #         stationIdx = synthDF["STATION"] == station
    #         synthDF.loc[monthIdx & stationIdx, "PRCP"] = prcpSample[:, m, s]            
    #         synthDF.loc[monthIdx & stationIdx, "TAVG"] = tavgSample[:, s] 
     
    return pd.DataFrame()
