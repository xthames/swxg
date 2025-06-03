# imports
import os
import sys
import pandas as pd
import numpy as np
import scipy as sp
import pickle


# filepaths
processedDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed"
cmip6Dir = os.path.dirname(os.path.dirname(__file__)) + r"/cmip6"


def GenerateBiasCorrections(): 
    # load in the NOAA bias-corrected observations we need
    try:
        noaaMonthly = pd.read_csv(processedDir + r"/NOAA/NOAA_UCRBMonthly.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Bias correction for CMIP6 data requires processed NOAA observations!")
    stations = sorted(set(noaaMonthly["NAME"]))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # which models are we looking at
    nasaDir, hydroDir = cmip6Dir + r"/nasa", cmip6Dir + r"/ornl" 
    nasaModels = [m for m in os.listdir(nasaDir + r"/historical") if os.path.isdir(nasaDir + r"/historical/{}".format(m))]
    hydroModels = [m for m in os.listdir(hydroDir + r"/historical") if os.path.isdir(hydroDir + r"/historical/{}".format(m))]
    maxCorrs, counter = 12*(len(nasaModels) + 4*len(hydroModels)), 0

    # build a dictionary for all of the bias correction factors, then convert that into a dataframe
    biascorrColumns = ["STATION", "MONTH", "SOURCE", "MODEL", "FORCING", "DOWNSCALE", "PRCP BIAS INFO", "TAVG BIAS INFO"]
    biascorrDict = {} 
    for station in stations:
        noaaStationIdx = noaaMonthly["NAME"] == station
        # nasa models...
        for model in nasaModels:
            counter += 1
            print("Generating bias corrections: {}% complete | station: {}, source: NASA, model: {}".format(int(float(counter)/maxCorrs*100.), station, model).ljust(125), end="\r")
            nasaPMDF = pd.read_csv(nasaDir + r"/historical/{}/Raw_NASACMIP6_historical_{}_Daily.csv".format(model, model))
            nasaPathwayIdx, nasaStationIdx = nasaPMDF["PATHWAY"] == "historical", nasaPMDF["STATION"] == station
            nasaEntry = nasaPMDF.loc[nasaPathwayIdx & (nasaPMDF["MODEL"] == model) & nasaStationIdx]
            nasaHistYears = sorted(set(nasaEntry["YEAR"].values))
            noaaHistYearsIdx = (noaaMonthly["YEAR"] >= min(nasaHistYears)) & (noaaMonthly["YEAR"] <= max(nasaHistYears))
            # for each month...
            for month in months:
                noaaMonthIdx = noaaMonthly["MONTH"] == month
                nasaMonthIdx = nasaEntry["MONTH"] == month
                nasaMonthlyPRCP, nasaMonthlyTAVG = [], []
                # aggregate monthly NASA data from daily
                for year in nasaHistYears:
                    nasaYearIdx = nasaEntry["YEAR"] == year
                    nasaMonthlyPRCP.append(np.nansum(nasaEntry.loc[nasaMonthIdx & nasaYearIdx, "PRCP"].values))
                    nasaMonthlyTAVG.append(np.nanmean(nasaEntry.loc[nasaMonthIdx & nasaYearIdx, "TAVG"].values))
                # NOAA monthly
                noaaMonthlyPRCP = noaaMonthly.loc[noaaStationIdx & noaaHistYearsIdx & noaaMonthIdx, "PRCP"].values
                noaaMonthlyTAVG = noaaMonthly.loc[noaaStationIdx & noaaHistYearsIdx & noaaMonthIdx, "TAVG"].values
                # -- quantile mapping (precipitation)
                noaaMonthlyPRCPNaNless = [val for val in noaaMonthlyPRCP if not np.isnan(val)]
                noaaECDF = sp.stats.ecdf(noaaMonthlyPRCPNaNless)
                noaaECDFi = sp.interpolate.interp1d(noaaECDF.cdf.probabilities, noaaECDF.cdf.quantiles, fill_value="extrapolate")
                nasaECDF = sp.stats.ecdf([val for val in nasaMonthlyPRCP if not np.isnan(val)])
                nasaPRCPMean = np.nanmean(nasaMonthlyPRCP)
                # -- variance scaling  (temperature)
                noaaTAVGMean, nasaTAVGMean = np.nanmean(noaaMonthlyTAVG), np.nanmean(nasaMonthlyTAVG) 
                stdTAVGScale = np.nanstd(noaaMonthlyTAVG) / np.nanstd(nasaMonthlyTAVG)
                # put them in the dictionary
                biascorrKey = (station, month, "NASA", model, "N/A", "N/A") 
                biascorrDict[biascorrKey] = {"PRCP BIAS INFO": [noaaECDFi, nasaECDF, nasaPRCPMean], "TAVG BIAS INFO": [noaaTAVGMean, stdTAVGScale, nasaTAVGMean]}
        # ornl models...
        for forcing in ["Daymet", "Livneh"]:
            for downscale in ["DBCCA", "RegCM"]:
                for model in hydroModels:
                    counter += 1
                    print("Generating bias corrections: {}% complete | station: {}, source: ORNL, model: {}".format(int(float(counter)/maxCorrs*100.), station, model).ljust(125), end="\r")
                    # loading the files...
                    hydroModelFiles = [f for f in os.listdir(hydroDir + r"/historical/{}".format(model)) if (forcing in f and downscale in f)]
                    prcpFile = [f for f in hydroModelFiles if "PRCP" in f][0]
                    hydroPRCPDF = pd.read_csv(hydroDir + r"/historical/{}/{}".format(model, prcpFile))
                    tminFile = [f for f in hydroModelFiles if "TMIN" in f][0]
                    tmaxFile = [f for f in hydroModelFiles if "TMAX" in f][0]
                    tminModelDF = pd.read_csv(hydroDir + r"/historical/{}/{}".format(model, tminFile))
                    tmaxModelDF = pd.read_csv(hydroDir + r"/historical/{}/{}".format(model, tmaxFile))
                    hydroTAVGDF = pd.concat([tminModelDF, tmaxModelDF["TMAX"]], axis=1)
                    hydroTAVGDF["TAVG"] = hydroTAVGDF[["TMIN", "TMAX"]].mean(axis=1)
                    # indexing...
                    hydroHistYears = sorted(set(hydroPRCPDF["YEAR"].values))
                    noaaHistYearsIdx = (noaaMonthly["YEAR"] >= min(hydroHistYears)) & (noaaMonthly["YEAR"] <= max(hydroHistYears)) 
                    hydroPRCPStationIdx = hydroPRCPDF["STATION"] == station 
                    hydroTAVGStationIdx = hydroTAVGDF["STATION"] == station
                    # for each month...
                    for month in months:
                        noaaMonthIdx = noaaMonthly["MONTH"] == month
                        hydroPRCPMonthIdx = hydroPRCPDF["MONTH"] == month 
                        hydroTAVGMonthIdx = hydroTAVGDF["MONTH"] == month
                        # noaa, hydro monthly
                        noaaMonthlyPRCP = noaaMonthly.loc[noaaStationIdx & noaaHistYearsIdx & noaaMonthIdx, "PRCP"].values
                        noaaMonthlyTAVG = noaaMonthly.loc[noaaStationIdx & noaaHistYearsIdx & noaaMonthIdx, "TAVG"].values
                        hydroMonthlyPRCP = hydroPRCPDF.loc[hydroPRCPStationIdx & hydroPRCPMonthIdx, "PRCP"].values
                        hydroMonthlyTAVG = hydroTAVGDF.loc[hydroTAVGStationIdx & hydroTAVGMonthIdx, "TAVG"].values
                        # -- quantile mapping (precipitation)
                        noaaMonthlyPRCPNaNless = [val for val in noaaMonthlyPRCP if not np.isnan(val)]
                        noaaECDF = sp.stats.ecdf(noaaMonthlyPRCPNaNless)
                        noaaECDFi = sp.interpolate.interp1d(noaaECDF.cdf.probabilities, noaaECDF.cdf.quantiles, fill_value="extrapolate")
                        hydroECDF = sp.stats.ecdf([val for val in hydroMonthlyPRCP if not np.isnan(val)])
                        hydroPRCPMean = np.nanmean(hydroMonthlyPRCP)
                        # -- variance scaling  (temperature)
                        noaaTAVGMean, hydroTAVGMean = np.nanmean(noaaMonthlyTAVG), np.nanmean(hydroMonthlyTAVG) 
                        stdTAVGScale = np.nanstd(noaaMonthlyTAVG) / np.nanstd(hydroMonthlyTAVG)
                        # put them in the dictionary 
                        biascorrKey = (station, month, "ORNL", model, forcing, downscale) 
                        biascorrDict[biascorrKey] = {"PRCP BIAS INFO": [noaaECDFi, hydroECDF, hydroPRCPMean], "TAVG BIAS INFO": [noaaTAVGMean, stdTAVGScale, hydroTAVGMean]}
    
    # save the dictionary as-is using pickle
    print("")
    with open(processedDir + r"/CMIP6/CMIP6_Biases.pkl", "wb") as f:
        pickle.dump(biascorrDict, f)


# aggregate all of the CMIP6 data to a single .csv, with bias-corrections applied
def BiasCorrectCMIP6():
    # setting up the dictionary, dataframe
    cmip6Dict = {}
    cmip6Columns = ["NAME", "YEAR", "MONTH", "PRCP", "TAVG"]
    
    # if the data is from the NASA source
    if source.upper() == "NASA":
        modelDir = cmip6Dir + r"/nasa/{}/{}".format(pathway, model)
        modelDF = pd.read_csv(modelDir + r"/Raw_NASACMIP6_{}_{}_Daily.csv".format(pathway, model))
        nasaYears = sorted(set(modelDF["YEAR"].values))
        # for each month...
        for month in months:
            nasaMonthIdx = modelDF["MONTH"] == month
            # for each station...
            for station in stations:
                nasaStationIdx = modelDF["STATION"] == station
                nasaPRCPMonthly = [np.nansum(modelDF.loc[nasaMonthIdx & nasaStationIdx & (modelDF["YEAR"] == year), "PRCP"].values) for year in nasaYears]
                biasKey = [k for k in cmip6BCDict.keys() if ((source.upper() in k) and (model in k) and (month in k) and (station in k))][0]
                # for each year...
                for year in nasaYears:
                    # aggregate the monthly data     
                    nasaYearIdx = modelDF["YEAR"] == year 
                    nasaPRCP = np.nansum(modelDF.loc[nasaYearIdx & nasaMonthIdx & nasaStationIdx, "PRCP"].values)
                    nasaTAVG = np.nanmean(modelDF.loc[nasaYearIdx & nasaMonthIdx & nasaStationIdx, "TAVG"].values)
                    # bias correct
                    # -- precipitation
                    nasaPRCPScale = cmip6BCDict[biasKey]["PRCP BIAS INFO"][2] / np.nanmean(nasaPRCPMonthly)
                    nasaPRCPProb = cmip6BCDict[biasKey]["PRCP BIAS INFO"][1].cdf.evaluate(nasaPRCPScale * nasaPRCP)
                    nasaPRCPbc = cmip6BCDict[biasKey]["PRCP BIAS INFO"][0](nasaPRCPProb) / nasaPRCPScale
                    nasaPRCPbc = 0. if nasaPRCPbc < 0. else nasaPRCPbc
                    # -- temperature
                    nasaTAVGbc = (nasaTAVG - cmip6BCDict[biasKey]["TAVG BIAS INFO"][2])*cmip6BCDict[biasKey]["TAVG BIAS INFO"][1] + cmip6BCDict[biasKey]["TAVG BIAS INFO"][0]  
                    # add to dictionary
                    cmip6DictKey = (source, pathway, model, "N/A", "N/A", year, month, station)
                    cmip6Dict[cmip6DictKey] = [station, year, month, nasaPRCPbc, nasaTAVGbc]
    # if the data is from the ORNL source
    else:
        modelDir = cmip6Dir + r"/ornl/{}/{}".format(pathway, model)
        # pulling in the right files...
        hydroModelFiles = [f for f in os.listdir(modelDir) if (forcing in f and downscale in f)]
        prcpFile = [f for f in hydroModelFiles if "PRCP" in f][0]
        ornlPRCPDF = pd.read_csv(modelDir + r"/{}".format(prcpFile), na_values="--")
        tminFile = [f for f in hydroModelFiles if "TMIN" in f][0]
        tmaxFile = [f for f in hydroModelFiles if "TMAX" in f][0]
        tminModelDF = pd.read_csv(modelDir + r"/{}".format(tminFile), na_values="--")
        tmaxModelDF = pd.read_csv(modelDir + r"/{}".format(tmaxFile), na_values="--")
        ornlTAVGDF = pd.concat([tminModelDF, tmaxModelDF["TMAX"]], axis=1)
        ornlTAVGDF["TAVG"] = ornlTAVGDF[["TMIN", "TMAX"]].mean(axis=1)
        ornlYears = sorted(set(ornlPRCPDF["YEAR"].values))
        # for each month...
        for month in months: 
            ornlPRCPMonthIdx = ornlPRCPDF["MONTH"] == month
            ornlTAVGMonthIdx = ornlTAVGDF["MONTH"] == month
            # for each station...
            for station in stations:
                ornlPRCPStationIdx = ornlPRCPDF["STATION"] == station
                ornlTAVGStationIdx = ornlTAVGDF["STATION"] == station
                ornlPRCPMonthly = [val for val in ornlPRCPDF.loc[ornlPRCPMonthIdx & ornlPRCPStationIdx, "PRCP"].values if not np.isnan(val)]
                biasKey = [k for k in cmip6BCDict.keys() if ((source.upper() in k) and (model in k) and (forcing in k) and (downscale in k) and (month in k) and (station in k))][0]
                # for each year...
                for year in ornlYears:
                    ornlPRCPYearIdx = ornlPRCPDF["YEAR"] == year
                    ornlTAVGYearIdx = ornlTAVGDF["YEAR"] == year
                    ornlPRCPEntry = ornlPRCPDF.loc[ornlPRCPYearIdx & ornlPRCPMonthIdx & ornlPRCPStationIdx]
                    ornlTAVGEntry = ornlTAVGDF.loc[ornlTAVGYearIdx & ornlTAVGMonthIdx & ornlTAVGStationIdx]
                    # bias correction
                    if ornlPRCPEntry.empty or np.isnan(ornlPRCPEntry["PRCP"].values[0]):
                        ornlPRCPbc = np.NaN
                    else:
                        # precipitation
                        ornlPRCP = ornlPRCPEntry["PRCP"].values[0]
                        ornlPRCPScale = cmip6BCDict[biasKey]["PRCP BIAS INFO"][2] / np.nanmean(ornlPRCPMonthly)
                        ornlPRCPProb = cmip6BCDict[biasKey]["PRCP BIAS INFO"][1].cdf.evaluate(ornlPRCPScale * ornlPRCP)
                        ornlPRCPbc = cmip6BCDict[biasKey]["PRCP BIAS INFO"][0](ornlPRCPProb) / ornlPRCPScale
                        ornlPRCPbc = 0. if ornlPRCPbc < 0. else ornlPRCPbc
                    if ornlTAVGEntry.empty or np.isnan(ornlTAVGEntry["TAVG"].values[0]):
                        ornlTAVGbc = np.NaN
                    else:
                        # temperature
                        ornlTAVG = ornlTAVGEntry["TAVG"].values[0]
                        ornlTAVGbc = (ornlTAVG - cmip6BCDict[biasKey]["TAVG BIAS INFO"][2])*cmip6BCDict[biasKey]["TAVG BIAS INFO"][1] + cmip6BCDict[biasKey]["TAVG BIAS INFO"][0]  
                    # add to dictionary
                    cmip6DictKey = (source, pathway, model, forcing, downscale, year, month, station)
                    cmip6Dict[cmip6DictKey] = [station, year, month, ornlPRCPbc, ornlTAVGbc]
    # convert from dictionary to dataframe
    cmip6DF = pd.DataFrame.from_dict(cmip6Dict, orient="index", columns=cmip6Columns)
    cmip6DF.reset_index(inplace=True, drop=True)
    cmip6DF.sort_values(by=["NAME", "YEAR"], inplace=True)
    cmip6DF.to_csv(processedDir + "/{}/{}_UCRBMonthly.csv".format(dataRepo, repoName), index=False)


# execute the main file
if __name__ == "__main__": 
    # identify the path to the historical and ssp data from the passed dataRepo
    dataRepo = sys.argv[1]
    repoName = dataRepo.replace("/", "_")
    pathPieces = dataRepo.split("/")
    source, pathway, model = pathPieces[1], pathPieces[2], pathPieces[3] 
    forcing = pathPieces[4] if "ornl" in dataRepo else ""
    downscale = pathPieces[5] if "ornl" in dataRepo else ""
    
    # load the bias correction dataframe
    with open(processedDir + r"/CMIP6/CMIP6_Biases.pkl", "rb") as f:
        cmip6BCDict = pickle.load(f)
    stations = sorted(set([k[0] for k in cmip6BCDict.keys()]))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # bias correct the CMIP6 data, generating per-path UCRBMonthly.csvs
    BiasCorrectCMIP6()


