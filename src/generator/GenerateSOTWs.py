# imports
import os
import sys
import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from copulas import univariate, bivariate
from statsmodels.tsa.ar_model import AutoReg
import warnings
import copy


# environment arguments
dataRepo = sys.argv[1]
nSOTW = int(sys.argv[2]) + 1


# filepaths
processedDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed"
syntheticDir = os.path.dirname(os.path.dirname(__file__)) + r"/synthetic"


# select the corresponding number of scenarios of GMMHMM/copulae information
def DiscoverSOTWs(): 
    # copy our observation DF and dailyMinT dict
    scenarioObsDF = noaaObsDF.copy()
    # -- remove data from before the first year/after the last year in the fitted NOAA data
    scenarioObsDF = scenarioObsDF[(scenarioObsDF["YEAR"] >= min(noaaFitYears)) & (scenarioObsDF["YEAR"] <= max(noaaFitYears))] 

    # extract the parameters from the LHC sample for this scenario
    sowParameters = sotwSamples[nSOTW-1]
    scenarioAnnualPMean, scenarioAnnualPStd = sowParameters[0], sowParameters[1] 
    scenarioAnnualHP = sowParameters[2]
    scenarioAnnualTMean, scenarioAnnualTStd = sowParameters[3], sowParameters[4]

    # -- GMMHMM --     
    for s, station in enumerate(stations):
        obsStationIdx = noaaObsDF["NAME"] == station
        scenarioStationIdx = scenarioObsDF["NAME"] == station
        # -- sample mean, std
        log10ScenarioMean = scenarioAnnualPMean * relativeProfileDict[nSOTW]["meanP"][s]
        log10ScenarioStd = scenarioAnnualPStd * relativeProfileDict[nSOTW]["stdP"][s]
        # -- update the scenario observation copy
        log10ObsMean, log10ObsStd = noaaGMMHMMDict[station]["means"][0], noaaGMMHMMDict[station]["stds"][0]
        stationObs = scenarioObsDF.loc[scenarioStationIdx, "PRCP"].values
        stationObs[stationObs <= 2.54E-4] = np.inf
        scenarioObsDF.loc[scenarioStationIdx, "PRCP"] = np.power(10., (np.log10(stationObs) - log10ObsMean)*(log10ScenarioStd/log10ObsStd) + log10ScenarioMean) 
    scenarioObsDF.replace(np.inf, 0., inplace=True)
    
    # develop the scenario's GMMHMM dictionary
    scenarioMultisiteDict = {}
    # -- format the dictionary for synthesizing readability
    precipDFDict = {}
    for year in noaaFitYears:
        pdfYearIdx = scenarioObsDF["YEAR"] == year
        pdfLine = []
        for station in stations:
            pdfStationIdx = scenarioObsDF["NAME"] == station
            pdfLine.append(np.nansum(scenarioObsDF.loc[pdfYearIdx & pdfStationIdx, "PRCP"].values)) 
        precipDFDict[year] = pdfLine
    scenarioMultisiteDict["precipDF"] = np.log10(pd.DataFrame().from_dict(precipDFDict, orient="index", columns=stations))
    # -- build the GMMHMM (really covarying Gaussians)
    scenarioNumStates = 1
    scenarioGMMHMM = GMMHMM(n_components=scenarioNumStates, n_iter=1000, covariance_type="full", init_params="cmw")
    scenarioGMMHMM.startprob_ = np.full(shape=(scenarioNumStates, scenarioNumStates), fill_value=1./scenarioNumStates)
    scenarioGMMHMM.transmat_ = np.full(shape=(scenarioNumStates, scenarioNumStates), fill_value=1./scenarioNumStates)
    scenarioGMMHMM.weights_ = np.full(shape=(scenarioNumStates, scenarioNumStates), fill_value=1./scenarioNumStates)
    scenarioGMMHMM.means_ = scenarioMultisiteDict["precipDF"].mean(axis=0).values.reshape(1, 1, len(stations))
    scenarioGMMHMM.covars_ = scenarioMultisiteDict["precipDF"].cov().values.reshape(1, 1, len(stations), len(stations))
    scenarioMultisiteDict["model"] = scenarioGMMHMM

    # -- TAVG -- 
    for m, month in enumerate(months):
        scenarioMonthIdx = scenarioObsDF["MONTH"] == month
        monthObs = scenarioObsDF.loc[scenarioMonthIdx, "TAVG"].values
        # -- sample mean, std
        scenarioTMean = scenarioAnnualTMean + relativeProfileDict[nSOTW]["meanT"][m] 
        scenarioTStd = scenarioAnnualTStd * relativeProfileDict[nSOTW]["stdT"][m] 
        # -- update the scenario observation copy 
        obsMean, obsStd = np.nanmean(noaaCopulaeDict[month]["TAVG"]), np.nanstd(noaaCopulaeDict[month]["TAVG"])
        scenarioObsDF.loc[scenarioMonthIdx, "TAVG"] = (monthObs - obsMean)*(scenarioTStd/obsStd) + scenarioTMean
         
    # develop the scenario's copulae dictionary
    scenarioJointPT = {month: {} for month in months} 
    noaaFullYears = range(min(noaaFitYears), max(noaaFitYears)+1)
    for m, month in enumerate(months):
        cMonthIndex = scenarioObsDF["MONTH"] == month
        # -- build the Frank copulae
        fCop = bivariate.Frank()
        fCop.theta = scenarioAnnualHP + relativeProfileDict[nSOTW]["hp"][m]
        # -- format the data for synthesizing readability
        scenarioJointPT[month]["CopulaDF"] = pd.DataFrame().from_dict({"Frank": fCop}, orient="index", columns=["Copula"])
        histPrcp, histTavg = [], []
        for year in noaaFullYears:
            cYearIndex = scenarioObsDF["YEAR"] == year
            cEntryPrcp = scenarioObsDF.loc[cMonthIndex & cYearIndex, "PRCP"]
            cEntryTavg = scenarioObsDF.loc[cMonthIndex & cYearIndex, "TAVG"]
            histPrcp.append(np.NaN if len(cEntryPrcp.values) == 0 else np.nanmean(cEntryPrcp.astype(float).values))
            histTavg.append(np.NaN if len(cEntryTavg.values) == 0 else np.nanmean(cEntryTavg.astype(float).values))
        histPrcp, histTavg = np.array(histPrcp), np.array(histTavg)
        histPrcp[np.isnan(histPrcp)], histTavg[np.isnan(histTavg)] = np.nanmean(histPrcp), np.nanmean(histTavg)
        scenarioJointPT[month]["PRCP"], scenarioJointPT[month]["TAVG"] = histPrcp, histTavg
        scenarioJointPT[month]["PRCP ARFit"] = AutoReg(scenarioJointPT[month]["PRCP"], lags=[1]).fit()
        scenarioJointPT[month]["TAVG ARFit"] = AutoReg(scenarioJointPT[month]["TAVG"], lags=[1]).fit() 
        prcpUnivariate, tavgUnivariate = univariate.Univariate(), univariate.Univariate()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            prcpUnivariate.fit(np.array(scenarioJointPT[month]["PRCP ARFit"].resid, dtype=float))
            tavgUnivariate.fit(np.array(scenarioJointPT[month]["TAVG ARFit"].resid, dtype=float))
        scenarioJointPT[month]["PRCP Resid Dist"] = prcpUnivariate
        scenarioJointPT[month]["TAVG Resid Dist"] = tavgUnivariate 
      
    # develop the scenario's adjusted daily temperatures
    scenarioDailyTDict = copy.deepcopy(noaaDailyTDict)
    for k in noaaDailyTDict.keys():
        station, year = k[0], k[1]
        dfStationIdx = scenarioObsDF["NAME"] == station
        dfYearIdx = scenarioObsDF["YEAR"] == year
        dfEntry = scenarioObsDF.loc[dfStationIdx & dfYearIdx]
        if dfEntry.empty:
            del scenarioDailyTDict[k]
            continue
        else:
            for month in months:
                dfMonthIdx = dfEntry["MONTH"] == month
                if any(scenarioDailyTDict[k][:, 0] == month):
                    histTAVG = np.nanmean(np.nanmean(scenarioDailyTDict[k][scenarioDailyTDict[k][:, 0] == month, 2:], axis=1))
                    scenarioTAVG = dfEntry.loc[dfMonthIdx, "TAVG"].values[0] 
                    scenarioDailyTDict[k][scenarioDailyTDict[k][:, 0] == month, 2] = scenarioDailyTDict[k][scenarioDailyTDict[k][:, 0] == month, 2] + (scenarioTAVG - histTAVG)
                    scenarioDailyTDict[k][scenarioDailyTDict[k][:, 0] == month, 3] = scenarioDailyTDict[k][scenarioDailyTDict[k][:, 0] == month, 3] + (scenarioTAVG - histTAVG) 

    # # test prints
    # print(noaaObsDF.loc[(noaaObsDF["YEAR"] >= min(noaaFullYears)) & (noaaObsDF["YEAR"] <= max(noaaFullYears))])
    # print(scenarioObsDF)
    # for m, month in enumerate(months):
    #     print(noaaCopulaeDict[month]["CopulaDF"].at["Frank", "params"])
    #     print(scenarioAnnualHP + relativeProfileDict[nSOTW]["hp"][m])
    # print(noaaDailyTDict[("Altenbern", 1950)])
    # print(scenarioDailyTDict[("Altenbern", 1950)])

    # save the scenario observations 
    scenarioObsDF.to_csv(syntheticDir + r"/{}/Scenario{}/{}_Scenario{}_UCRBMonthly.csv".format(dataRepo, nSOTW, dataRepo, nSOTW), index=False)
    np.save(syntheticDir + r"/{}/Scenario{}/{}_Scenario{}_MultisiteCGMCs.npy".format(dataRepo, nSOTW, dataRepo, nSOTW), scenarioMultisiteDict)
    np.save(syntheticDir + r"/{}/Scenario{}/{}_Scenario{}_Copulae.npy".format(dataRepo, nSOTW, dataRepo, nSOTW), scenarioJointPT)
    np.save(syntheticDir + r"/{}/Scenario{}/{}_Scenario{}_DailyTs.npy".format(dataRepo, nSOTW, dataRepo, nSOTW), scenarioDailyTDict)


# running the program
if __name__ == "__main__": 
    # load in the NOAA observations
    noaaObsDF = pd.read_csv(processedDir + r"/NOAA/NOAA_UCRBMonthly.csv")
    stations = sorted(set(noaaObsDF["NAME"]))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] 
    
    # load in the noaa fits
    noaaGMMHMMDict = np.load(processedDir + r"/NOAA/NOAA_MultisiteGMMHMMFit.npy", allow_pickle=True).item()
    noaaCopulaeDict = np.load(processedDir + r"/NOAA/NOAA_CopulaFits.npy", allow_pickle=True).item()
    noaaDailyTDict = np.load(processedDir + r"/NOAA/NOAA_UCRBDailyT.npy", allow_pickle=True).item()
    noaaFitYears = noaaGMMHMMDict["precipDF"].index.values
    # load in the LHC SOTW samples, profiles
    sotwDict = np.load(syntheticDir + r"/{}/SOTWs.npy".format(dataRepo), allow_pickle=True).item()
    sotwSamples = sotwDict["sows"]
    relativeProfileDict = sotwDict["profiles"]
    # create the SOTWs
    DiscoverSOTWs()



