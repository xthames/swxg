# import
import os
import sys
from multiprocessing import Process
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import random
from hmmlearn.hmm import GMMHMM
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
import warnings
import math
from scipy import integrate
from scipy.stats import rankdata, mannwhitneyu
from scipy import optimize
from NOAAClimateDataReader import CalculateGeographicDistance
from statsmodels.tools import eval_measures
from copulas import univariate, bivariate, multivariate
import copulae


# environment variable
dataRepo = sys.argv[1]
repoName = dataRepo.replace("/", "_")

# filepaths
processedDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed/{}".format(dataRepo)
plotsDir = os.path.dirname(os.path.dirname(__file__)) + r"/plots"


# aggregate the raw monthly DFs into a singular .csv
def AggregateCSV(filenameConvention):
    # get the filenames based on the naming convention
    fileNames = [f for f in os.listdir(processedDir) if filenameConvention in f] 
    aggDF = pd.DataFrame()
    for fileName in fileNames:
        # read in the file, add it to the aggDF
        uniqueDF = pd.read_csv(processedDir + "/" + fileName)
        aggDF = pd.concat([aggDF if not aggDF.empty else None, uniqueDF], ignore_index=True)
        # remove the singular file
        os.remove(processedDir + "/" + fileName)
    aggDF.to_csv(processedDir + "/" + filenameConvention + ".csv", index=False, mode="w")


# aggregate the raw monthly dicts into a singular .npy
def AggregateNPY(filenameConvention):
    # get the filenames based on the naming convention
    fileNames = [f for f in os.listdir(processedDir) if filenameConvention in f]
    aggDict = {}
    for fileName in fileNames:
        # read in the file, add it to the aggDict
        stationDict = np.load(processedDir + "/" + fileName, allow_pickle=True).item()
        for k, v in stationDict.items():
            aggDict[k] = v
        # remove the singular file
        os.remove(processedDir + "/" + fileName)
    np.save(processedDir + "/" + filenameConvention + ".npy", aggDict)


# managing function for fitting precip data to GMMHMM
def FitGMMHMM(monthlyUCRB):
    # format the data: create annual data, log10-scale them
    annualDict, monthlyDict = FormatPrecipData(monthlyUCRB)

    # use the formatted data to fit the multisite GMMHM
    FitMultisiteGMMHMM(annualDict, monthlyDict)
        

# format the data (a single time) for use in the GMMHMM
def FormatPrecipData(data):
    # stations and years
    stations, years = sorted(set(data["NAME"])), sorted(set(data["YEAR"]))[:-1]
    months = sorted(set(data["MONTH"]), key=lambda x: dt.datetime.strptime(x, "%b"))
    nStations, nYears, nMonths = len(stations), len(years), len(months)    

    # create a matrix to hold the time-sequence of monthly and annual precipitations for each station
    seqMonthlyPrecip = np.full(shape=(nYears * nMonths, nStations), fill_value=np.NaN)
    seqAnnualPrecip = np.full(shape=(nYears, nStations), fill_value=np.NaN)
    for s, station in enumerate(stations):
        stationIdx = data["NAME"] == station
        for y, year in enumerate(years):
            yearIdx = data["YEAR"] == year
            yearEntry = data.loc[stationIdx & yearIdx,"PRCP"]
            yearValue = np.nansum(data.loc[stationIdx & yearIdx, "PRCP"].values)
            if yearValue == 0:
                yearValue = np.NaN
            monthValues = np.full(shape=nMonths, fill_value=np.NaN)
            for m, month in enumerate(months):
                monthIdx = data["MONTH"] == month
                monthValue = data.loc[stationIdx & yearIdx & monthIdx, "PRCP"].values
                if monthValue.size == 0:
                    monthValue = np.NaN
                else:
                    monthValue = monthValue[0]
                monthValues[m] = monthValue
            seqAnnualPrecip[y, s] = yearValue
            seqMonthlyPrecip[y*nMonths:(y+1)*nMonths, s] = monthValues

    # identifying where data is missing (NaN indices) for both monthly and annual data
    seqMonthlyDF = pd.DataFrame(seqMonthlyPrecip, columns=stations, index=[(yr, mo) for yr in years for mo in months])
    seqAnnualDF = pd.DataFrame(seqAnnualPrecip, columns=stations, index=years)
    monthlyNaNs, annualNaNs = np.any(seqMonthlyDF.isna(), axis=1).values, np.any(seqAnnualDF.isna(), axis=1).values

    # helper function to determine the length of each sequence for the fit
    def DetermineSequenceLengths(inputNaNBools):
        seqList, run = [], 0
        for ii in range(len(inputNaNBools)):
            if inputNaNBools[ii]:
                seqList.append(run)
                run = 0
            else:
                run += 1
        seqList.append(run)
        return np.array([entry for entry in seqList if entry != 0], dtype=int)

    # sequence lengths to use
    annualSeqLengths = DetermineSequenceLengths(annualNaNs)
    monthlySeqLengths = DetermineSequenceLengths(monthlyNaNs)

    # drop NaNs, transform the annual data with log10
    seqMonthlyDF.dropna(inplace=True)
    seqAnnualDF.dropna(inplace=True)
    transformedAnnualDF = np.log10(seqAnnualDF)

    # return the data
    return {"df": transformedAnnualDF, "lengths": annualSeqLengths}, {"df": seqMonthlyDF, "lengths": monthlySeqLengths}


# fit the multisite GMMHMM
def FitMultisiteGMMHMM(annualDict, monthlyDict):
    # the number of components/states to use for the GMMHMM
    numStates = ModelComponentSelector(annualDict, minStates=1, maxStates=4) if ("NOAA" in dataRepo) else 1
    # data from the function inputs
    transformedDF, seqLengths = annualDict["df"], annualDict["lengths"]
    # station, month info
    stations = sorted(set(transformedDF.columns))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    nStations, nMonths = len(stations), len(months)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # fit a multidimensional Gaussian mixture HMM to data
    # -- note: the covergence EM method can nortoriously can fall into local optima
    # -- suggested fix: run a few fits, take the one with the highest log-likelihood
    # -- when checking states: use AIC and BIC criteria
    bestModel, bestLL, bestSeed = None, None, None
    for _ in range(20):
        # define our hidden Markov model, whose states' covariance must be positive definite
        positiveDefinite, tempSeed, tempModel = False, None, None
        while not positiveDefinite:
            # get the random state
            tempSeed = random.getstate()

            # define the parameters for the model
            tempModelObject = GMMHMM(n_components=numStates, n_iter=1000, covariance_type="full", init_params="cmw")
            tempModelObject.startprob_ = np.full(shape=numStates, fill_value=1./numStates)
            tempModelObject.transmat_ = np.full(shape=(numStates, numStates), fill_value=1./numStates)
            tempModel = tempModelObject.fit(transformedDF.values, lengths=seqLengths)

            # get, reshape the covariance matrices
            tempCovars = tempModel.covars_.reshape((numStates, nStations, nStations))

            # check that each state has a covariance matrix that is positive definite, meaning:
            # -- it's symmetric
            symmetricCheck = all([np.allclose(tempCovars[i].T, tempCovars[i]) for i in range(numStates)])
            # -- it's eigenvalue are all positive
            eigCheck = all([(np.linalg.eigvalsh(tempCovars[i]) > 0).all() for i in range(numStates)])

            # confirm positive definiteness of covariance matrix
            positiveDefinite = symmetricCheck and eigCheck

        # calculate the loglikelihood of this model
        tempScore = tempModel.score(transformedDF.values, lengths=seqLengths)

        # get the model with the highest score, set that as the best one for each test of states
        if not bestLL or tempScore > bestLL:
            bestLL = tempScore
            bestModel = tempModel
            bestSeed = tempSeed
    seed = bestSeed
    model = bestModel
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # reorder the moments matrices to match least wet --> the wettest states
    def ConceptualReorder(modelMeans, modelCovars, modelTransmat):
        # check which conceptual order the states should be in: state 0 driest, ... , state n wettest
        conceptualOrderIndices = np.argsort([np.sum(modelMeans[i]) for i in range(len(modelMeans))])
        if np.all(np.diff(conceptualOrderIndices) >= 0):
            return modelMeans, modelCovars, modelTransmat

        # reorder means, covariances
        reorderedMeans = modelMeans[conceptualOrderIndices, :]
        reorderedCovars = modelCovars[conceptualOrderIndices, :]

        # reorder transmat
        reorderedTransmat = np.full(shape=modelTransmat.shape, fill_value=np.NaN)
        for i in range(modelTransmat.shape[0]):
            reorderedTransmat[conceptualOrderIndices[i], :] = modelTransmat[i, conceptualOrderIndices]

        # return the reordered matrices
        return reorderedMeans, reorderedCovars, reorderedTransmat
    model.means_, model.covars_, model.transmat_ = ConceptualReorder(model.means_, model.covars_, model.transmat_)

    # predict the states that each of the data are in
    hiddenStates = model.predict(transformedDF.values, lengths=seqLengths)

    # transform/reshape means, covars for dictionary
    means = model.means_.reshape((model.n_components, nStations))
    maskedCovars = model.covars_.copy()
    maskedCovars[maskedCovars < 0] = np.NaN
    stds = np.sqrt(maskedCovars).reshape((model.n_components, nStations, nStations))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # create a dictionary to hold this best-of GMMHMM information
    multisiteDict = {station: {} for station in stations}
    # -- means and stds, by station
    for s, station in enumerate(stations):
        stationMeans, stationStds = np.array([]), np.array([])
        for n in range(model.n_components):
            stationMeans = np.append(stationMeans, means[n][s])
            stationStds = np.append(stationStds, np.diag(stds[n])[s])
        multisiteDict[station]["means"] = stationMeans
        multisiteDict[station]["stds"] = stationStds

    # -- the dataframe of monthly values (not log10), annual values (yes log10)
    multisiteDict["monthlyDF"] = monthlyDict["df"]
    multisiteDict["precipDF"] = transformedDF

    # -- the actual GMMHMM object
    # noinspection PyTypeChecker
    multisiteDict["seed"] = seed
    # noinspection PyTypeChecker
    multisiteDict["model"] = model

    # -- hidden states, transition probabilities
    multisiteDict["hiddenStates"] = hiddenStates
    multisiteDict["tProbs"] = model.transmat_
    
    # -- bestfit the annual station correlations with distance, find the residuals
    if (model.n_components == 1) and ("NASA" in dataRepo): 
        modelCorrMatrix = np.full(shape=(12, 12), fill_value=np.NaN)
        for i, stationy in enumerate(stations):
            for j, stationx in enumerate(stations):
                modelCorrMatrix[i][j] = model.covars_[0][0][i][j] / (multisiteDict[stationx]["stds"][0] * multisiteDict[stationy]["stds"][0])
        modelCorrDF = pd.DataFrame(modelCorrMatrix, index=stations, columns=stations)
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        ds, corrs = [], []
        dDF = pd.DataFrame(index=stations, columns=stations)
        for s1, station1 in enumerate(stations):
            station1Idx = monthlyDF["NAME"] == station1
            dists = []
            for s2, station2 in enumerate(stations):
                if s2 >= s1: continue
                station2Idx = monthlyDF["NAME"] == station2
                lat1, lon1 = list(set(monthlyDF.loc[station1Idx, "LAT"].values))[0], list(set(monthlyDF.loc[station1Idx, "LON"].values))[0]
                lat2, lon2 = list(set(monthlyDF.loc[station2Idx, "LAT"].values))[0], list(set(monthlyDF.loc[station2Idx, "LON"].values))[0]
                elev1, elev2 = list(set(monthlyDF.loc[station1Idx, "ELEV"].values))[0], list(set(monthlyDF.loc[station2Idx, "ELEV"].values))[0]
                dist = CalculateGeographicDistance([lat1, lon1, elev1], [lat2, lon2, elev2])
                dDF.at[station1, station2] = dist
                ds.append(dist)
                corrs.append(modelCorrDF.at[station1, station2])
        popts = optimize.curve_fit(exp_decay, np.array(ds)/100., corrs)
        resids = np.array(corrs) - exp_decay(np.array(ds), popts[0][0], popts[0][1]/100., popts[0][2])
        multisiteDict["corrdist"] = {"expDecayParams": [popts[0][0], popts[0][1]/100., popts[0][2]], "resids": resids, "distanceDF": dDF} 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # save the dictionary
    # noinspection PyTypeChecker
    np.save(processedDir + r"/{}_MultisiteGMMHMMFit.npy".format(repoName), multisiteDict)


# checking which number of components fits the GMMHMM best
def ModelComponentSelector(annualDict, minStates=1, maxStates=5):
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


# managing function for fitting precip/temp data to monthly copulae
def FitCopulae(rawMonthlyDF, multisiteDict):
    # explore the dependence between the monthly precipitation and temperature
    ExplorePTDependence(rawMonthlyDF, multisiteDict["precipDF"].index.values)

    # actually fit the copulae with the p/T data
    ConstructCopulae(rawMonthlyDF, multisiteDict["precipDF"].index.values)


# explore the dependence between precipitation and temperature
def ExplorePTDependence(pTData, hmmYears):
    # get the stations, months
    stations = sorted(set(pTData["NAME"]))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    completeYears = [y for y in range(min(hmmYears), max(hmmYears) + 1)]

    # spatially averaged correlation between precipitation and temperature
    spatialCorrDF = pd.DataFrame({"Kendall": np.NaN, "Spearman": np.NaN}, index=months)
    spatialDataDict = {month: {"PRCP": [], "TAVG": []} for month in months}
    for month in months:
        monthIndex = pTData["MONTH"] == month
        for year in completeYears:
            yearIndex = pTData["YEAR"] == year
            spatialDataDict[month]["PRCP"].append(np.nanmean(pTData.loc[monthIndex & yearIndex, "PRCP"].values))
            spatialDataDict[month]["TAVG"].append(np.nanmean(pTData.loc[monthIndex & yearIndex, "TAVG"].values))
        spatialCorrDF.at[month, "Kendall"] = pd.DataFrame({"PRCP": spatialDataDict[month]["PRCP"], "TAVG": spatialDataDict[month]["TAVG"]}).corr(method="kendall")["PRCP"]["TAVG"]
        spatialCorrDF.at[month, "Spearman"] = pd.DataFrame({"PRCP": spatialDataDict[month]["PRCP"], "TAVG": spatialDataDict[month]["TAVG"]}).corr(method="spearman")["PRCP"]["TAVG"]
    
    # kendall/spearman correlation metric plot
    saCorrFig, axis = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    saCorrFig.suptitle("Correlation of Spatially-Averaged {} P/T Data by Month".format(repoName))
    saCorrFig.supxlabel("Month"), saCorrFig.supylabel("Correlation Coefficient [-]")
    axis.grid()
    axis.set_ylim(-1, 1)
    axis.set_xticks(range(len(months)))
    axis.set_xticklabels(months, rotation=45)
    axis.hlines(0, xmin=0, xmax=11, colors="black", linestyles="dashed")
    axis.plot(range(len(months)), spatialCorrDF["Kendall"], marker="o", label=r"Kendall $\tau$")
    axis.plot(range(len(months)), spatialCorrDF["Spearman"], marker="o", label=r"Spearman $\rho$")
    axis.legend()
    plt.tight_layout()
    saCorrFig.savefig(plotsDir + r"/copulae/{}/{}_SpatialAveragePTCorrExplorationByMonth.svg".format(dataRepo, repoName))
    plt.close()
     
    # plot scatterplot of spatially averaged precipitation and temperature
    pTDistFig = plt.figure(figsize=(14, 9))
    subFigs = pTDistFig.subfigures(3, 4)
    for i, subFig in enumerate(subFigs.flat):
        axes = subFig.subplots(2, 2, gridspec_kw={"width_ratios": [4, 1], "height_ratios": [1, 3]})
        subFig.subplots_adjust(wspace=0, hspace=0)
        month = months[i]
        for j, axis in enumerate(axes.flat):
            if j == 0:
                axis.hist(spatialDataDict[month]["PRCP"], density=True, color="black")
                axis.set(xticks=[], yticks=[])
            if j == 1:
                axis.axis("off")
                axis.text(0.5, 0.5, month, transform=axis.transAxes, va="center", ha="center")
            if j == 2:
                axis.scatter(spatialDataDict[month]["PRCP"], spatialDataDict[month]["TAVG"], marker="o", facecolors="none", edgecolors="black")
            if j == 3:
                axis.hist(spatialDataDict[month]["TAVG"], density=True, color="black", orientation="horizontal")
                axis.set(xticks=[], yticks=[])
    # plt.tight_layout()
    pTDistFig.savefig(plotsDir + r"/copulae/{}/{}_SpatialAveragePTDistributionByMonth.svg".format(dataRepo, repoName))
    plt.close()
    

# fit a copula to the data by first pretreating it, then visually inspecting it
def ConstructCopulae(pTData, hmmYears, genScenarios=False):
    # general info
    stations, years = sorted(set(pTData["NAME"])), sorted(set(pTData["YEAR"]))
    if not genScenarios:
        completeYears = [y for y in range(min(hmmYears), max(hmmYears)+1)]
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    else:
        completeYears = hmmYears
        months = list(set(pTData["MONTH"].values))

    # extracting just the stations, years, months, precip, temp from the raw dataframe
    df = pd.DataFrame(columns=["STATION", "YEAR", "MONTH", "PRCP", "TAVG"])
    df["STATION"] = np.repeat(stations, len(completeYears)*len(months))
    df["YEAR"] = list(np.repeat(completeYears, len(months))) * len(stations)
    df["MONTH"] = months * (len(completeYears)*len(stations))
    for station in stations:
        stationIdx = pTData["NAME"] == station
        dfStationIdx = df["STATION"] == station
        for year in completeYears:
            yearIdx = pTData["YEAR"] == year
            dfYearIdx = df["YEAR"] == year
            if genScenarios:
                histPrcp, histTavg = pTData.loc[stationIdx & yearIdx, "PRCP"].values, pTData.loc[stationIdx & yearIdx, "TAVG"].values
                df.loc[dfStationIdx & dfYearIdx, "PRCP"] = histPrcp if len(histPrcp) > 0 else np.NaN
                df.loc[dfStationIdx & dfYearIdx, "TAVG"] = histTavg if len(histTavg) > 0 else np.NaN
            else:
                for month in months:
                    monthIdx = pTData["MONTH"] == month
                    dfMonthIdx = df["MONTH"] == month
                    histPrcp, histTavg = pTData.loc[stationIdx & yearIdx & monthIdx, "PRCP"].values, pTData.loc[stationIdx & yearIdx & monthIdx, "TAVG"].values
                    df.loc[dfStationIdx & dfYearIdx & dfMonthIdx, "PRCP"] = histPrcp if len(histPrcp) > 0 else np.NaN
                    df.loc[dfStationIdx & dfYearIdx & dfMonthIdx, "TAVG"] = histTavg if len(histTavg) > 0 else np.NaN

    # establishing a dictionary to hold everything, filling missing values from the group average
    pTDict = {month: {"PRCP": [], "TAVG": []} for month in months}
    for month in pTDict:
        monthIdx = df["MONTH"] == month
        histPrcp, histTavg = [], []
        for year in completeYears:
            yearIdx = df["YEAR"] == year
            stationAveragedEntry = df.loc[monthIdx & yearIdx]
            histPrcp.append(np.NaN if stationAveragedEntry.empty else np.nanmean(df.loc[monthIdx & yearIdx, "PRCP"].astype(float).values))
            histTavg.append(np.NaN if stationAveragedEntry.empty else np.nanmean(df.loc[monthIdx & yearIdx, "TAVG"].astype(float).values))
        histPrcp, histTavg = np.array(histPrcp), np.array(histTavg)
        histPrcp[np.isnan(histPrcp)], histTavg[np.isnan(histTavg)] = np.nanmean(histPrcp), np.nanmean(histTavg)
        pTDict[month]["PRCP"], pTDict[month]["TAVG"] = histPrcp, histTavg

    # pretreating the data
    # -- autocorrelation
    pTDict = InvestigateAutocorrelation(pTDict, lag=1, genScenarios=genScenarios)
    # -- stationarity
    if not genScenarios:
        InvestigateStationarity(pTDict, numGroups=2)

    # visually inspecting the data
    # -- K-plots
    if not genScenarios:
        VisualizeDependenceStructure(pTDict)
    # -- fit copulae, model ranking criteria
    jointPT = FitCopulaFamilies(pTDict, genScenarios=genScenarios)

    # save the fitted copulas
    # noinspection PyTypeChecker
    if not genScenarios:
        np.save(processedDir + r"/{}_CopulaFits.npy".format(repoName), jointPT)
    else:
        return jointPT


# investigating autocorrelation in the p/T data
def InvestigateAutocorrelation(ptDict, lag, genScenarios=False):
    # find the autoregressive fit of [LAG] for each station's month
    for month in ptDict.keys():
        histPrcp, histTavg = ptDict[month]["PRCP"], ptDict[month]["TAVG"]
        ptDict[month]["PRCP ARFit"] = AutoReg(histPrcp, lags=[lag]).fit()
        ptDict[month]["TAVG ARFit"] = AutoReg(histTavg, lags=[lag]).fit()

        if (not genScenarios) or (len(ptDict.keys()) > 1):
            # find the underlying univariate distribution from those residuals
            prcpUnivariate, tavgUnivariate = univariate.Univariate(), univariate.Univariate()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                prcpUnivariate.fit(np.array(ptDict[month]["PRCP ARFit"].resid, dtype=float))
                tavgUnivariate.fit(np.array(ptDict[month]["TAVG ARFit"].resid, dtype=float))
            ptDict[month]["PRCP Resid Dist"] = prcpUnivariate
            ptDict[month]["TAVG Resid Dist"] = tavgUnivariate
    
    if not genScenarios:
        # plot the ACF of the raw data and residual autocorrelation
        weatherVars = ["PRCP", "TAVG"]
        for weatherVar in weatherVars:
            weatherColor = "royalblue" if weatherVar == "PRCP" else "firebrick"
            ACFPlot, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all", sharey="all")
            ACFPlot.suptitle("{} {} ACF from AR{}".format(repoName, weatherVar, lag))
            ACFPlot.supxlabel("Lag [-]"), ACFPlot.supylabel("ACF [-]")
            months = list(ptDict.keys())
            for i, axis in enumerate(axes.flat):
                axis.grid()
                # actual plot of the ACF
                plot_acf(ax=axis, x=ptDict[months[i]][weatherVar], color=weatherColor, vlines_kwargs={"color": weatherColor, "label": None})
                plot_acf(ax=axis, x=ptDict[months[i]][weatherVar + " ARFit"].resid, color="black", vlines_kwargs={"color": "grey", "label": None})
                axis.set(title=months[i])
            plt.tight_layout()
            ACFPlot.savefig(plotsDir + r"/copulae/{}/{}_Autocorrelation_{}.svg".format(dataRepo, repoName, weatherVar))
            plt.close()

    # return the new dictionary
    return ptDict


# investigating stationarity in the p/T data
def InvestigateStationarity(ptDict, numGroups):
    numGroups = 2 if numGroups < 2 else numGroups
    # plot the ACF of the raw data and residual autocorrelation
    weatherVars = ["PRCP", "TAVG"]
    barWidth = 1 / numGroups
    for weatherVar in weatherVars:
        MWUPlot, axis = plt.subplots(nrows=1, ncols=1, figsize=(14, 9))
        MWUPlot.suptitle("{} Residuals Stationarity Check for {} with {} Groups".format(repoName, weatherVar, numGroups))
        MWUPlot.supxlabel("Month"), MWUPlot.supylabel("Mann-Whitney U p-Value [-]")
        months = list(ptDict.keys())
        mwuPValues = np.full(shape=(len(months), numGroups-1), fill_value=np.NaN)
        for m, month in enumerate(months):
            resids = ptDict[month][weatherVar + " ARFit"].resid
            groupChunk = len(resids)//numGroups if len(resids) % numGroups == 0 else len(resids)//numGroups+1
            dataGroups = []
            for n in range(numGroups):
                dataGroups.append(resids[n*groupChunk:(n+1)*groupChunk])
            for g in range(numGroups-1):
                mwuPValues[m, g] = mannwhitneyu(x=np.array(dataGroups[g], dtype=float), y=np.array(dataGroups[g+1], dtype=float), nan_policy="omit")[1]

        # formatting, actually plotting
        axis.grid()
        axis.set(ylim=[0, 1])
        for g in range(numGroups-1):
            axis.bar([x+g*barWidth for x in range(len(months))], mwuPValues[:, g], width=barWidth, zorder=10)
        axis.hlines(0.05, -1, 12, color="black", linestyles="dashed", zorder=11)
        axis.set_xticks([m+(numGroups-2)*(barWidth/2) for m in range(len(months))])
        axis.set_xticklabels(labels=months, rotation=45)
        plt.tight_layout()
        MWUPlot.savefig(plotsDir + r"/copulae/{}/{}_Stationarity_{}_{}Groups.svg".format(dataRepo, repoName, weatherVar, numGroups))
        plt.close()


# visualizing the dependence structure of the copulas via K-plot
def VisualizeDependenceStructure(ptDict):
    # define the functional form of the integrand for W_i:n
    def W_inIntegrand(w, idx, num):
        scale = num * math.comb(num - 1, idx - 1)
        u = w - (w * np.log(w))
        return scale * w * u ** (idx - 1) * (1 - u) ** (num - idx) * -np.log(w)

    # plot K-plot for each station
    kPlots, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all", sharey="all")
    kPlots.suptitle("{} Monthly K-Plots".format(repoName))
    kPlots.supxlabel("$W_{i:n}$"), kPlots.supylabel("$H_{(i)}$")
    months = list(ptDict.keys())

    # for each month...
    for i, axis in enumerate(axes.flat):
        n = len(ptDict[months[i]]["PRCP ARFit"].resid)
        H_i, W_in = np.array([]), np.array([])

        # calculate W_in, H_i
        for ii in range(n):
            p_i = ptDict[months[i]]["PRCP ARFit"].resid[ii]
            T_i = ptDict[months[i]]["TAVG ARFit"].resid[ii]
            if np.isnan(p_i) or np.isnan(T_i):
                continue
            H = (1 / (n - 1)) * sum([1 for j in range(n) if j != ii and
                                     ptDict[months[i]]["PRCP ARFit"].resid[j] <= p_i and
                                     ptDict[months[i]]["TAVG ARFit"].resid[j] <= T_i])
            H_i = np.append(H_i, H)
            W_in = np.append(W_in, integrate.quad(W_inIntegrand, a=0, b=1, args=(ii + 1, n))[0])
        H_i.sort()

        # plot H_(i) against W_in
        axis.grid()
        axis.set_title(months[i])
        axis.plot([jj / n for jj in range(n)], [jj / n for jj in range(n)], c="black", linestyle="dashed")
        axis.plot(W_in, H_i, c="magenta")
    plt.tight_layout()
    kPlots.savefig(plotsDir + r"/copulae/{}/{}_KPlots.svg".format(dataRepo, repoName))
    plt.close()


# fitting the various copula families and then comparing them
def FitCopulaFamilies(ptDict, genScenarios=False):
    # getting the CDF of the empirical copula for bootstrapping goodness-of-fit metrics
    def CalculateEmpiricalCopulaCDFatPoint(pseudoObservations, point):
        nData = pseudoObservations.shape[0]
        vecCn = np.full(shape=nData, fill_value=0.)
        for k in range(nData):
            U1, U2 = pseudoObservations[k, 0], pseudoObservations[k, 1]
            if U1 <= point[0] and U2 <= point[1]:
                vecCn[k] = 1.
        return np.sum(vecCn) / nData

    # generating the bootstrapping metrics
    def BootstrapCopulaCVMandKS(pseudoObservations, theoryCopula, fromDF=False):
        # need to have more samples than the number of observations to bootstrap
        n = pseudoObservations.shape[0]
        if fromDF:
            columnNames = theoryCopula.to_dict()["columns"]

        # Cramér Von-Mises (S_n), Kolmogorov-Smirnov (T_n)
        SnElements = np.full(shape=n, fill_value=np.NaN)
        TnElements = np.full(shape=n, fill_value=np.NaN)
        for k in range(n):
            po = pseudoObservations[k, :]
            C_n = CalculateEmpiricalCopulaCDFatPoint(pseudoObservations, po)
            Bstar_m = theoryCopula.cdf(pd.DataFrame(data={columnNames[0]: [po[0]], columnNames[1]: [po[1]]})) if fromDF else theoryCopula.cdf(np.atleast_2d(po))
            Bstar_m = Bstar_m[0] if type(Bstar_m) in [list, np.ndarray] else Bstar_m
            SnElements[k] = (C_n - Bstar_m)**2.
            TnElements[k] = np.abs(C_n  - Bstar_m)
        S_n = np.sum(SnElements)
        T_n = np.max(TnElements)

        # return the metrics
        return S_n, T_n

    # determine the best-fitting copula for the data by looking at AIC, Sn, Tn
    def FindBestCopula(copulaDF):
        # copying the dataframe so we can sort things out
        aicSorted, snSorted, tnSorted = copulaDF.copy().sort_values(by=["AIC"]), copulaDF.copy().sort_values(by=["S_n"]), copulaDF.copy().sort_values(by=["T_n"])

        # choosing the best copula
        winningFamilies = [aicSorted.index.values[0], snSorted.index.values[0], tnSorted.index.values[0]]
        nBestFamilies = len(set(winningFamilies))
        if nBestFamilies == 1 or nBestFamilies == 3:
            # -- if all three metrics are the lowest for a single family, use that one
            # -- if each family claims one lowest metric, just use AIC
            # -- if AIC is at least 2 less each other family, that's a better model
            return [copulaDF.at[aicSorted.index.values[0], "Copula"], aicSorted.index.values[0]]
        else:
            # pick the family where both S_n and T_n are winning if true, otherwise pick the winning AIC family
            if len({snSorted.index.values[0], tnSorted.index.values[0]}) == 1:
                return [copulaDF.at[winningFamilies[1], "Copula"], winningFamilies[1]]
            else:
                return [copulaDF.at[winningFamilies[0], "Copula"], winningFamilies[0]]

    # include the pseudo-observations in the dictionary
    for month in ptDict.keys():
        nP, nT = len(ptDict[month]["PRCP ARFit"].resid), len(ptDict[month]["TAVG ARFit"].resid)
        ptDict[month]["PRCP pObs"] = rankdata(ptDict[month]["PRCP ARFit"].resid, method="average", nan_policy="omit") / (nP+1)
        ptDict[month]["TAVG pObs"] = rankdata(ptDict[month]["TAVG ARFit"].resid, method="average", nan_policy="omit") / (nT+1)

    # fit copulas! from the packages I've found, the best options (for capturing negative dependencies) are:
    # -- independence, frank, gaussian
    families = ["Independence", "Frank", "Gaussian"]
    copulaFitDict = {month: pd.DataFrame(data={"Copula": [None] * len(families),
                                               "params": [np.NaN] * len(families),
                                               "AIC": [np.Inf] * len(families),
                                               "S_n": [[]] * len(families),
                                               "T_n": [[]] * len(families)},
                                         index=families) for month in ptDict}

    # start the fitting loop over stations, months
    minAIC, maxAIC = np.Inf, -np.Inf
    for month in ptDict:
        # psuedo-observations as an array
        pseudoObs = np.array([ptDict[month]["PRCP pObs"], ptDict[month]["TAVG pObs"]]).T

        # Independence
        iCop = copulae.IndepCopula()
        copulaFitDict[month].at["Independence", "Copula"] = iCop
        copulaFitDict[month].at["Independence", "params"] = np.NaN
        copulaFitDict[month].at["Independence", "AIC"] = eval_measures.aic(llf=iCop.log_lik(pseudoObs),
                                                                           nobs=pseudoObs.size, df_modelwc=np.array([]).size)
        if not genScenarios:
            iS_n, iT_n = BootstrapCopulaCVMandKS(pseudoObs, iCop)
            copulaFitDict[month].at["Independence", "S_n"] = iS_n
            copulaFitDict[month].at["Independence", "T_n"] = iT_n

        # Frank
        fCop = bivariate.Frank()
        fCop.fit(pseudoObs)
        copulaFitDict[month].at["Frank", "Copula"] = fCop
        copulaFitDict[month].at["Frank", "params"] = fCop.theta
        copulaFitDict[month].at["Frank", "AIC"] = eval_measures.aic(llf=np.sum(fCop.log_probability_density(pseudoObs)),
                                                                    nobs=pseudoObs.size, df_modelwc=np.array(fCop.theta).size)
        if not genScenarios:
            fS_n, fT_n = BootstrapCopulaCVMandKS(pseudoObs, fCop)
            copulaFitDict[month].at["Frank", "S_n"] = fS_n
            copulaFitDict[month].at["Frank", "T_n"] = fT_n

        # Gaussian
        pseudoObsDF = pd.DataFrame(data=pseudoObs, columns=["uP", "uT"])
        gCop = multivariate.GaussianMultivariate(distribution={"uP": univariate.UniformUnivariate(),
                                                               "uT": univariate.UniformUnivariate()})
        gCop.fit(pseudoObsDF)
        copulaFitDict[month].at["Gaussian", "Copula"] = gCop
        copulaFitDict[month].at["Gaussian", "params"] = gCop.correlation["uP"]["uT"]
        gCopulae = copulae.GaussianCopula()
        gCopulae.params = gCop.correlation["uP"]["uT"]
        copulaFitDict[month].at["Gaussian", "AIC"] = eval_measures.aic(llf=gCopulae.log_lik(data=pseudoObsDF.values, to_pobs=False),
                                                                       nobs=pseudoObs.size, df_modelwc=np.array(gCop.correlation["uP"]["uT"]).size)
        if not genScenarios:
            gS_n, gT_n = BootstrapCopulaCVMandKS(pseudoObs, gCop, fromDF=True)
            copulaFitDict[month].at["Gaussian", "S_n"] = gS_n
            copulaFitDict[month].at["Gaussian", "T_n"] = gT_n

        # bounds for the AIC
        for family in families:
            if copulaFitDict[month].at[family, "AIC"] < minAIC:
                minAIC = copulaFitDict[month].at[family, "AIC"]
            if copulaFitDict[month].at[family, "AIC"] > maxAIC:
                maxAIC = copulaFitDict[month].at[family, "AIC"]

        # add the copula families dictionary to the ptDict, for conciseness
        ptDict[month]["CopulaDF"] = copulaFitDict[month]

        # find and add the best copula to the ptDict
        if not genScenarios:
            ptDict[month]["BestCopula"] = FindBestCopula(copulaFitDict[month])

    # return the copula fits
    return ptDict


# actually run the script
if __name__ == "__main__":
    # helper function for multiprocessing
    def MultiprocessHelper(fns, fninputs):
        pross = []
        for i, fn in enumerate(fns):
            p = Process(target=fn, args=fninputs[i])
            p.start()
            pross.append(p)
        for p in pross:
            p.join()
    
    # when looking at the NOAA data
    if "NOAA" in dataRepo:
        # aggregate the per station data NOAA .csvs, .npys
        aggregationFnInputList = [("Raw{}_Monthly".format(repoName),), ("Raw{}_Biases".format(repoName),), 
                                  ("{}_UCRBDailyT".format(repoName),), ("{}_UCRBMonthly".format(repoName),)]
        # -- remove the aggregated files if they already exist --> multiprocessing apparently APPENDS to .csvs ???
        for fileConvention in aggregationFnInputList:
            fp = processedDir + "/" + fileConvention[0] + ".csv"
            if os.path.isfile(fp):
                os.remove(fp)
        aggregationFnList = [AggregateCSV, AggregateCSV, AggregateNPY, AggregateCSV]
        MultiprocessHelper(aggregationFnList, aggregationFnInputList)
     
    # WAIT FOR ALL OF THOSE PROCESSES TO FINISH BEFORE MOVING ON 
    # -- run the GMMHMM constructor
    monthlyDF = pd.read_csv(processedDir + r"/{}_UCRBMonthly.csv".format(repoName))
    FitGMMHMM(monthlyDF)
    # -- run the Copula constructor 
    GMMHMMDict = np.load(processedDir + r"/{}_MultisiteGMMHMMFit.npy".format(repoName), allow_pickle=True).item()
    FitCopulae(monthlyDF, GMMHMMDict)

