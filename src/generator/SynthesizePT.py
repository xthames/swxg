# import
import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import rankdata
import copy


# filepaths
processedDir = os.path.dirname(os.path.dirname(__file__)) + r"/processed"
syntheticDir = os.path.dirname(os.path.dirname(__file__)) + r"/synthetic"
origPrcFP = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU/COclim2015.prc"
origTemFP = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU/COclim2015.tem"
origFdFP = os.path.dirname(os.path.dirname(__file__)) + r"/cdss-dev/cm2015_StateCU/StateCU/COclim2015.fd"


# create an appropriate number of RNG seeds so that we can access unique seeds when script is executed in parallel
def GenerateRandomSeeds(dataRepos, nScns, nSims):
    for nScn in range(nScns): 
        ss = np.random.SeedSequence()
        childSeeds = ss.spawn(nSims)
        rng_list = [np.random.default_rng(childSeed) for childSeed in childSeeds]
        # noinspection PyTypeChecker
        np.save(syntheticDir + r"/{}/Scenario{}/{}_Scenario{}_SynthPT_RNGs.npy".format(dataRepos, nScn + 1, dataRepos.replace("/", "_"), nScn + 1), rng_list)


# k-NN disaggregation algorithm from Apipattanavis et al. (2007), Nowak et al. (2010), and Quinn (2020 supp.)
def kNNPrecipDisaggregation(sampleData, historicMonthlyData, historicYearlyData, completeYears, k=None):
    # (0) convert to real-space if samples are in log-space
    stations, years = sorted(set(historicMonthlyData["NAME"])), sorted(set(historicYearlyData.index.values))
    months = sorted(set(historicMonthlyData["MONTH"]), key=lambda x: dt.datetime.strptime(x, "%b"))
    synthData = 10**sampleData

    # (1) create k, weights
    # -- recommended to be int(sqrt(n)), where n is the number of years in the time-series (Lall & Sharma, 1996)
    k = round(np.sqrt(len(years))) if not k else k
    # -- make the weights
    w = np.array([(1 / j) for j in range(1, k+1)]) / sum([(1 / j) for j in range(1, k+1)])

    # (2) spatial averages for historic and synth
    histSpatialAvg = np.nanmean(10**historicYearlyData.values, axis=1)
    synthSpatialAvg = np.nanmean(synthData, axis=1)

    # (3) link the summed historic year with the year itself, empty vector to fill with year choices
    # noinspection PyTypeChecker
    yearHistPair = np.reshape([[years[i], histSpatialAvg[i]] for i in range(len(years))], newshape=(len(years), 2))
    kNNSelectedYears = np.full(shape=len(completeYears), fill_value=np.NaN)

    # -- create 3D array to hold the disaggregated monthly data
    # -- row: years, col: months, z: stations
    disaggregatedSample = np.full(shape=(len(completeYears), len(months), len(stations)), fill_value=np.NaN)

    # for each year in the synthetic data...
    for j, sumSynthYear in enumerate(synthSpatialAvg):
        # (4) calculate Euclidean distance (Manhattan distance for 1D) between individual synthetic and all historical
        # noinspection PyTypeChecker
        yearSynthDist = np.reshape([[yearHistPair[i, 0], abs(sumSynthYear - yearHistPair[i, 1])] for i in range(len(years))], newshape=(len(years), 2))
        # -- ascending sort the years based on distance, only consider first k years
        sortedYearDist = yearSynthDist[yearSynthDist[:, 1].argsort()]
        # (5) choose which year from the set of years using pre-determined weights
        kNNSelectedYears[j] = rng.choice(sortedYearDist[:k, 0], p=w)

    # (6) maintain spatial proportionality: synth_{month}/synth_{year} = hist_{month}/hist_{year}
    # -- thus, synth{month} = synth_{year} * (hist_{station month}/hist_{year})
    for i, year in enumerate(completeYears):
        yearDFIdx = historicMonthlyData["YEAR"] == kNNSelectedYears[i]
        for s, station in enumerate(stations):
            stationIdx = historicMonthlyData["NAME"] == station
            kNNSelectedMonthlyValues = historicMonthlyData.loc[stationIdx & yearDFIdx, "PRCP"].values
            disaggregatedSample[i, :, s] = synthData[i, s] * (kNNSelectedMonthlyValues / sum(kNNSelectedMonthlyValues))

    # return the disaggregated sample
    return disaggregatedSample


# function for using generated monthly precip and conditionally pairing that with a synthesized temperature
def PTPairGenerator(prcpSample, prcpDict, copulaDict):
    # conditional simulation of CDF(uT | uP)
    def SimulateConditionalUT(poP, copList):
        # extract the copula and its name
        copObj, copName = copList[0], copList[1]

        # different methodologies of conditional simulation based on different families
        match copName:
            case "Independence":
                # v = d/du [C(u,v)] --> since C(u,v) = u*v, marginal v *is* the inverse of the conditional CDF
                poT = rng.random(size=len(poP))
            case "Frank":
                # v = inverse of the conditional CDF -- c(v|u)^{-1} -- so the ppf of the copula given u
                y = rng.random(size=len(poP))
                try:
                    poT = copObj.percent_point(y, poP)
                except ValueError:
                    y = rng.random(size=len(poP))
                    poT = copObj.percent_point(y, poP)
            case "Gaussian":
                # conditional sampling starts with the Cholesky decomposition of the Gaussian parameter
                A = np.tril(np.linalg.cholesky(copObj.correlation.values))

                # transform to normal distribution for poP, generate on normal distribution for y
                normP = scipy.stats.norm.ppf(poP)
                y = scipy.stats.norm.ppf(rng.random(size=len(poP)))

                # matrix multiply to get sample simulation given input
                condSamp = A @ np.array([normP, y])

                # temperature marginals are the cdf of the temperature half of the conditional sample
                poT = scipy.stats.norm.cdf(condSamp[1])

        return poT

    def kNNTavgDisaggregation(sampleData, historicMonthlyData, mnth, k=None):
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
    
    # get some necessary parameters from each dictionary
    stations, gmmhmmyears = prcpDict["precipDF"].columns.values, prcpDict["precipDF"].index.values
    years = [y for y in range(min(gmmhmmyears), max(gmmhmmyears)+1)]
    months = list(copulaDict.keys())
    nYears, nMonths, nStations = prcpSample.shape

    # create the dictionary to hold all the dataframes
    synthDF = pd.DataFrame(columns=["STATION", "YEAR", "MONTH", "PRCP", "TAVG"])
    synthDF["STATION"] = np.repeat(stations, nYears * nMonths)
    synthDF["YEAR"] = list(np.repeat(years, nMonths)) * nStations
    synthDF["MONTH"] = months * (nYears * nStations)

    # for each month in the sample...
    for m, month in enumerate(months):
        monthIdx = synthDF["MONTH"] == month
        # the spatially-averaged precipitation data we're interested in
        saSynthPRCP = prcpSample[:, m, :].mean(axis=1)
        
        # transform the synthetic preciptation to residuals using the ARfit used in the copulas
        # -- note: first fittedvalues index is NaN, so fill that position with an average value from the rest of the fitted
        nP = len(saSynthPRCP)
        fullPrecipFitted = np.array([np.nanmean(copulaDict[month]["PRCP ARFit"].fittedvalues), *copulaDict[month]["PRCP ARFit"].fittedvalues])
        resid = saSynthPRCP - fullPrecipFitted

        # transform into uniform marginals
        uP = rankdata(resid, method="average", nan_policy="omit") / (nP+1)

        # conditional simulation of the uT | uP --> coming from {d/d(uP) [C(uP, uT)]}^{-1}
        # uT = SimulateConditionalUT(uP, copulaDict[station][month]["BestCopula"])
        # -- JUST USE FRANK
        uT = SimulateConditionalUT(uP, copList=[copulaDict[month]["CopulaDF"].at["Frank", "Copula"], "Frank"])

        # transform from marginals to residuals (using CDF^{-1}) to data (using AR fit)
        # -- same trick of averaging the fitted values for filling that earliest NaN point
        synthTResids = copulaDict[month]["TAVG Resid Dist"].ppf(uT)
        fullTavgFitted = np.array([np.nanmean(copulaDict[month]["TAVG ARFit"].fittedvalues), *copulaDict[month]["TAVG ARFit"].fittedvalues])
        saSynthTAVG = synthTResids + fullTavgFitted

        # sometimes the conditionally simulated temperature values are WAY too high or low
        # -- like, hotter than water boiling or colder than absolute zero
        # -- if this happens, resample the conditional temperatures until it doesn't happen
        histTavg = copulaDict[month]["TAVG"].astype(float)
        histTavgDiff = np.abs(np.nanmax(histTavg) - np.nanmin(histTavg))
        while np.any(saSynthTAVG < np.nanmin(histTavg) - histTavgDiff) or np.any(saSynthTAVG > np.nanmax(histTavg) + histTavgDiff):
            # uT = SimulateConditionalUT(uP, copulaDict[station][month]["BestCopula"])
            uT = SimulateConditionalUT(uP, copList=[copulaDict[month]["CopulaDF"].at["Frank", "Copula"], "Frank"])
            synthTResids = copulaDict[month]["TAVG Resid Dist"].ppf(uT)
            saSynthTAVG = synthTResids + fullTavgFitted

        # take the spatially-averaged temperatures and disaggregate to return per-station values
        tavgSample = kNNTavgDisaggregation(saSynthTAVG, monthlyDF, month)
        
        # for StateCU... (finding this has made me partially insane)
        # >> cold August temperatures (colder than July) are causing StateCU to occasionally crash
        # >>> specifically affecting stations: Grand Lake, Kremmling, Meredith, Yampa
        # >>>> more specifically, parcels: 3800545, 3800651, 5000517, 5200559, 5000653_D, 5300555_D
        if month == "Aug":
            problemStations = ["Grand Lake", "Kremmling", "Meredith", "Yampa"]
            # -- don't let the August temperatures for these stations be colder than July and less than 50 degF (minimum bound on synthetics)
            for s, station in enumerate(stations): 
                if station not in problemStations: continue
                # -- check for all the stations
                julFTemps = (synthDF.loc[(synthDF["MONTH"] == "Jul") & (synthDF["STATION"] == station), "TAVG"].astype(float).values)*(9./5.) + 32.
                augFTemps = (tavgSample[:, s])*(9./5.) + 32.
                if np.any((julFTemps > augFTemps) & (augFTemps < 50.)):
                    scuIdx = ((julFTemps > augFTemps) & (augFTemps < 50.))
                    stdDiffFTemps = np.std(julFTemps - augFTemps)  
                    for b in range(len(scuIdx)):
                        if scuIdx[b]:
                            augFTempResid = rng.normal(loc=0., scale=stdDiffFTemps) 
                            newAugFTemp = julFTemps[b] + abs(augFTempResid)
                            tavgSample[b, s] = (newAugFTemp - 32.)*(5./9.)

        # store precip, temp in the DF 
        for s, station in enumerate(stations):
            stationIdx = synthDF["STATION"] == station
            synthDF.loc[monthIdx & stationIdx, "PRCP"] = prcpSample[:, m, s]            
            synthDF.loc[monthIdx & stationIdx, "TAVG"] = tavgSample[:, s] 
     
    # return the synthDF
    return synthDF


# write the synthetic precipitation and temperature data to the appropriate .prc/.tem file
def WritePrecipAndTemp():
    stationDict = {"Altenbern": "USC00050214", "Collbran": "USC00051741",
                   "Eagle County": "USW00023063", "Fruita": "USC00053146",
                   "Glenwood Springs": "USC00053359", "Grand Junction": "USC00053489",
                   "Grand Lake": "USC00053500", "Green Mt Dam": "USC00053592",
                   "Kremmling": "USC00054664", "Meredith": "USC00055507",
                   "Rifle": "USC00057031", "Yampa": "USC00059265"}
    monthNumDict = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}

    # load in the data formatting lines from the StateCU .prc file
    prcDataFormatLine, temDataFormatLine = "", ""
    with (open(origPrcFP, "r") as prcFile, open(origTemFP, "r") as temFile):
        prcLines = prcFile.readlines()
        temLines = temFile.readlines()
    # #>EndHeader line tells us how StateCU will read in the data
    for l, line in enumerate(prcLines):
        if "#>EndHeader" in line:
            prcBreakLine = prcLines[l+1]
            prcColumnHeaderLine = prcLines[l+2]
            prcDataFormatLine = prcLines[l+3]
            prcUnitsLine = prcLines[l+4]
            break
    prcUnitsLine = prcUnitsLine.replace("1950", str(min(sorted(set(syntheticPTDF["YEAR"])))))
    prcUnitsLine = prcUnitsLine.replace("2013", str(max(sorted(set(syntheticPTDF["YEAR"])))))
    for l, line in enumerate(temLines):
        if "#>EndHeader" in line:
            temBreakLine = temLines[l+1]
            temColumnHeaderLine = temLines[l+2]
            temDataFormatLine = temLines[l+3]
            temUnitsLine = temLines[l+4]
            break
    temUnitsLine = temUnitsLine.replace("1950", str(min(sorted(set(syntheticPTDF["YEAR"])))))
    temUnitsLine = temUnitsLine.replace("2013", str(max(sorted(set(syntheticPTDF["YEAR"])))))
    # make a list of the b/e indices
    prcBs, prcEs, temBs, temEs = [0], [prcDataFormatLine.find("e")], [0], [temDataFormatLine.find("e")]
    prcB, prcE, temB, temE = prcBs[0], prcEs[0], temBs[0], temEs[0]
    while prcB >= 0 and prcE >= 0:
        prcB, prcE = prcDataFormatLine.find("b", prcE + 1), prcDataFormatLine.find("e", prcE + 1)
        prcBs.append(prcB), prcEs.append(prcE)
    while temB >= 0 and temE >= 0:
        temB, temE = temDataFormatLine.find("b", temE + 1), temDataFormatLine.find("e", temE + 1)
        temBs.append(temB), temEs.append(temE)
    prcBs, prcEs, temBs, temEs = prcBs[:-1], prcEs[:-1], temBs[:-1], temEs[:-1]

    # helper function for spacing out the data appropriately when writing
    def DataFormatter(val, b, e, frontPad=True):
        numMaxChars = e - b
        numDataChars = len(val)
        formattedData = ""
        if frontPad:
            for _ in range(numMaxChars - numDataChars + 1):
                formattedData += " "
            formattedData += val
        else:
            formattedData += val
            for _ in range(numMaxChars - numDataChars):
                formattedData += " "
        return formattedData

    # actually writing the files
    with (open(outputPrcFP, "w") as prcpFile, open(outputTemFP, "w") as tempFile):
        # define which simulation to use
        synthData = syntheticPTDF

        # into, sub-header stuff for the .prc and .tem files
        prcpFile.write(prcBreakLine)
        prcpFile.write(prcColumnHeaderLine)
        prcpFile.write(prcDataFormatLine)
        prcpFile.write(prcUnitsLine)
        tempFile.write(temBreakLine)
        tempFile.write(temColumnHeaderLine)
        tempFile.write(temDataFormatLine)
        tempFile.write(temUnitsLine)

        # start reading in the data in the appropriate order
        for year in sorted(set(synthData["YEAR"])):
            yearIdx = synthData["YEAR"] == year
            for station in sorted(set(synthData["STATION"])):
                stationIdx = synthData["STATION"] == station
                # fundamental line to write, with year and station ID
                prcpLine, tempLine = "", ""
                prcpLine += DataFormatter(str(year), prcBs[0], prcEs[0]) + " "
                prcpLine += DataFormatter(stationDict[station], prcBs[1], prcEs[1], frontPad=False) + " "
                tempLine += DataFormatter(str(year), temBs[0], temEs[0]) + " "
                tempLine += DataFormatter(stationDict[station], temBs[1], temEs[1], frontPad=False) + " "

                # add in the monthly data
                # -- precipitation
                prcpVals = synthData.loc[yearIdx & stationIdx, "PRCP"].astype(float).values
                prcpVals = prcpVals / 0.0254
                for p, prcpVal in enumerate(prcpVals):
                    prcpLine += DataFormatter(str(-999.00), prcBs[2 + p], prcEs[2 + p]) if np.isnan(prcpVal) else DataFormatter("{:.2f}".format(prcpVal), prcBs[2 + p], prcEs[2 + p])
                prcpLine += DataFormatter(str(-999.00), prcBs[2 + p + 1], prcEs[2 + p + 1]) + "\n" if np.all(np.isnan(prcpVals)) else DataFormatter("{:.2f}".format(np.nansum(prcpVals)), prcBs[2 + p + 1], prcEs[2 + p + 1]) + "\n"
                prcpFile.write(prcpLine)

                # -- temperature
                tempVals = synthData.loc[yearIdx & stationIdx, "TAVG"].astype(float).values
                tempVals = tempVals * (9. / 5.) + 32.
                for t, tempVal in enumerate(tempVals):
                    tempLine += DataFormatter(str(-999.00), temBs[2 + t], temEs[2 + t]) if np.isnan(tempVal) else DataFormatter("{:.2f}".format(tempVal), temBs[2 + t], temEs[2 + t])
                tempLine += DataFormatter(str(-999.00), temBs[2 + t + 1], temEs[2 + t + 1]) + "\n" if np.all(np.isnan(tempVals)) else DataFormatter("{:.2f}".format(np.nanmean(tempVals)), temBs[2 + t + 1], temEs[2 + t + 1]) + "\n"
                tempFile.write(tempLine)


# write the frost date file to a .fd file, generating a k-NN disaggregated daily synthetic temperature
def WriteFrostDate():
    stationDict = {"Altenbern": "USC00050214", "Collbran": "USC00051741",
                   "Eagle County": "USW00023063", "Fruita": "USC00053146",
                   "Glenwood Springs": "USC00053359", "Grand Junction": "USC00053489",
                   "Grand Lake": "USC00053500", "Green Mt Dam": "USC00053592",
                   "Kremmling": "USC00054664", "Meredith": "USC00055507",
                   "Rifle": "USC00057031", "Yampa": "USC00059265"}
    monthNumDict = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}

    # use the synthetic temperatures to shift the minT dailys
    def UpdateDailyT(dataDF, biasCorrDailyDict):
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        updatedTDict = copy.deepcopy(biasCorrDailyDict)
        
        for k in biasCorrDailyDict.keys():
            station, year = k[0], k[1]
            dfStationIdx = dataDF["STATION"] == station
            dfYearIdx = dataDF["YEAR"] == year
            dfEntry = dataDF.loc[dfStationIdx & dfYearIdx]
            if dfEntry.empty:
                del updatedTDict[k]
                continue
            else:
                for month in months:
                    dfMonthIdx = dfEntry["MONTH"] == month
                    if any(updatedTDict[k][:, 0] == month):
                        histTAVG = np.nanmean(np.nanmean(updatedTDict[k][updatedTDict[k][:, 0] == month, 2:], axis=1))
                        synthTAVG = dfEntry.loc[dfMonthIdx, "TAVG"].values[0] 
                        updatedTDict[k][updatedTDict[k][:, 0] == month, 2] = updatedTDict[k][updatedTDict[k][:, 0] == month, 2] + (synthTAVG - histTAVG)
                        updatedTDict[k][updatedTDict[k][:, 0] == month, 3] = updatedTDict[k][updatedTDict[k][:, 0] == month, 3] + (synthTAVG - histTAVG)
        return updatedTDict
    
    # disaggregate the synthetic temperature data to produce frost dates
    def kNNFrostDateDisaggregation(dataDF, noaaMonthlyDF, biasCorrDailyDict, k=None):
        # (0) outlining stations, years, months
        stations, years = sorted(set(dataDF["STATION"])), sorted(set(dataDF["YEAR"]))

        # (1) generating the weights
        k = round(np.sqrt(len(years))) if not k else k
        w = np.array([(1 / j) for j in range(1, k + 1)]) / sum([(1 / j) for j in range(1, k + 1)])

        # (2) spatially averaging the synthetic and historic *yearly* temperatures
        # -- not looking at monthly temperature, since there might be some correlation between yearly average temperature and frost dates
        synthSpatialTAVG, histSpatialTAVG = np.full(shape=(len(years), 2), fill_value=np.NaN), np.full(shape=(len(years), 2), fill_value=np.NaN)
        for y, year in enumerate(years):
            synthSpatialTAVG[y, 0], histSpatialTAVG[y, 0] = year, year
            synthYearlyTAVG, histYearlyTAVG = np.array([]), np.array([])
            synthYearIdx = dataDF["YEAR"] == year
            histYearIdx = noaaMonthlyDF["YEAR"] == year
            for station in stations:
                synthStationIdx = dataDF["STATION"] == station
                histStationIdx = noaaMonthlyDF["NAME"] == station
                synthYearlyTAVG = np.append(synthYearlyTAVG, np.nanmean(dataDF.loc[synthStationIdx & synthYearIdx, "TAVG"].astype(float).values))
                histTAVGSet = noaaMonthlyDF.loc[histStationIdx & histYearIdx, "TAVG"].astype(float).values
                histYearlyTAVG = np.NaN if len(histTAVGSet) == 0 else np.append(histYearlyTAVG, np.nanmean(histTAVGSet))
            synthSpatialTAVG[y, 1] = np.nanmean(synthYearlyTAVG)
            histSpatialTAVG[y, 1] = np.nanmean(histYearlyTAVG)

        kNNSelectedYears = np.full(shape=len(years), fill_value=np.NaN)
        for i, synthTAVG in enumerate(synthSpatialTAVG[:, 1]):
            # (3) find the differences between the average synth yearly and average hist yearly
            # noinspection PyTypeChecker
            tavgDiff = np.reshape([[synthSpatialTAVG[j, 0], np.abs(synthTAVG - histSpatialTAVG[j, 1])] for j in range(len(years))], newshape=(len(years), 2))
            # (4) ascending sort the years based on distance
            sortedYearDist = tavgDiff[tavgDiff[:, 1].argsort()]
            # (5) choose which fist-k years from the set of years using pre-determined weights
            kNNSelectedYears[i] = rng.choice(sortedYearDist[:k, 0], p=w)

        # ** using a (new) statistical methodology for disaggregating frost date ** 
        # (6) convert historic tmins at spring/fall 28/32 degF to DOYs
        doyDict = {station: {"ls28": [], "ls32": [], "ff32": [], "ff28": []} for station in stations}
        for i in range(len(kNNSelectedYears)):
            synthYearIdx = dataDF["YEAR"] == years[i]
            histYearIdx = noaaMonthlyDF["YEAR"] == int(kNNSelectedYears[i])
            for station in stations:
                synthStationIdx = dataDF["STATION"] == station
                histStationIdx = noaaMonthlyDF["NAME"] == station 
                synthAvgYear = np.nanmean(dataDF.loc[synthStationIdx & synthYearIdx, "TAVG"].astype(float).values)
                histAvgSet = noaaMonthlyDF.loc[histStationIdx & histYearIdx, "TAVG"].astype(float).values
                histAvgYear = np.nanmean(noaaMonthlyDF.loc[histStationIdx, "TAVG"].astype(float).values) if len(histAvgSet) == 0 else np.nanmean(histAvgSet)
                bcDailyDictKey = (station, int(kNNSelectedYears[i]))
                doyDictStationKeys = list(doyDict[station].keys())
                if bcDailyDictKey not in biasCorrDailyDict.keys():
                    # (*) formulate an average an entry if the raw data doesn't exist for that (station, year)
                    validKeys = [(station, int(kNNSelectedYears[j])) for j in range(len(kNNSelectedYears)) if (station, int(kNNSelectedYears[j])) in biasCorrDailyDict.keys()]                  
                    fillDict = {}
                    for validKey in validKeys:
                        validMDTm = biasCorrDailyDict[validKey].copy()
                        for v in range(validMDTm.shape[0]):
                            validMonth, validDay, validTmin = validMDTm[v, 0], validMDTm[v, 1], validMDTm[v, 2]
                            if validMonth == "Feb" and int(validDay) == 29:
                                continue
                            if (validMonth, validDay) not in fillDict.keys():
                                fillDict[(validMonth, validDay)] = [validMonth, validDay, [validTmin]]
                            else:
                                fillDict[(validMonth, validDay)][2].append(validTmin)
                    for k in fillDict.keys():
                        fillDict[k][2] = np.nanmean(fillDict[k][2])
                    monthDayTminDF = pd.DataFrame.from_dict(fillDict, orient="index", columns=["MONTH", "DAY", "TMIN"])
                    # (*) sort from Jan1 to Dec31 (excluding leap days)
                    dtStr = monthDayTminDF.apply(lambda x: str(int(kNNSelectedYears[i])) + "-" + monthNumDict[x["MONTH"]] + "-" + "{:02d}".format(x["DAY"]), axis=1)
                    doys = np.array([pd.Period(dtstr, freq="D").day_of_year for dtstr in dtStr.values]) - 1
                    monthDayTminDF.index = doys
                    monthDayTminDF = monthDayTminDF.sort_index()
                    monthDayTmin = monthDayTminDF.values
                else:
                    monthDayTmin = biasCorrDailyDict[bcDailyDictKey].copy() 
                shiftedTmin = monthDayTmin[:, 2] + synthAvgYear - histAvgYear
                fTmin = shiftedTmin * (9. / 5) + 32.
                # (*) search the shifted historic data for first instances when temperatures are below 32, 28 degF, find those indices
                below28, below32 = fTmin <= 28, fTmin <= 32
                spr28, fall28 = pd.Series(below28)[:len(below28) // 2], pd.Series(below28)[len(below28) // 2:]
                spr32, fall32 = pd.Series(below32)[:len(below32) // 2], pd.Series(below32)[len(below32) // 2:]
                ls28i = spr28[::-1].idxmax()
                ls32i = spr32[::-1].idxmax()
                ff32i = fall32.idxmax()
                ff28i = fall28.idxmax()
                # (*) convert the indices to DOY, and populate the dictionary
                for ii, idx in enumerate([ls28i, ls32i, ff32i, ff28i]):
                    doyStr = str(int(kNNSelectedYears[i])) + "-" + monthNumDict[monthDayTmin[idx, 0]] + "-" + "{:02d}".format(monthDayTmin[idx, 1])
                    doy = pd.Period(doyStr, freq="D").day_of_year
                    doyDict[station][doyDictStationKeys[ii]].append(doy) 

        # (7) sample from those doys, with caveats: doyLS28 <= doyLS32 < doyFF32 <= doyFF28 
        fdDict = {}
        for year in years:
            for station in stations:
                fdDictKey = (year, station)                
                # (*) QC -- make sure there's a clear divide between spring (up to 30 Jun) and fall (from 1 Jul and on)
                validDOYLS28 = [doy for doy in doyDict[station]["ls28"] if doy < 182]
                validDOYLS32 = [doy for doy in doyDict[station]["ls32"] if doy < 182] 
                validDOYFF32 = [doy for doy in doyDict[station]["ff32"] if doy > 182] 
                validDOYFF28 = [doy for doy in doyDict[station]["ff28"] if doy > 182] 
                # (*) QC -- LS32 should be >= LS28, FF32 should be > LS32, FF28 should be >= FF32
                doyLS28 = rng.choice(validDOYLS28)
                doyLS32 = rng.choice([doy if doy > doyLS28 else doyLS28 for doy in validDOYLS32])
                doyFF32 = rng.choice([doy for doy in validDOYFF32 if doy > doyLS32])
                doyFF28 = rng.choice([doy if doy > doyFF32 else doyFF32 for doy in validDOYFF28]) 
                # (*) QC: resample if...
                # -- the total length of the SP28-FA28 growing season is less than three weeks 
                # -- >> StateCU averages partial months by looking to closest month critical point (start/midpoint) -- can't have the spring/fall reference be the same date [~1 Jul]
                # -- FF32 is after 15 Dec
                # -- FF28 is after 31 Dec
                while (doyFF28 - doyLS28 < 21) or (doyFF32 >= 349) or (doyFF28 > 365):
                    doyLS28 = rng.choice(validDOYLS28)
                    doyLS32 = rng.choice([doy if doy > doyLS28 else doyLS28 for doy in validDOYLS32])
                    doyFF32 = rng.choice([doy for doy in validDOYFF32 if doy > doyLS32])
                    doyFF28 = rng.choice([doy if doy > doyFF32 else doyFF32 for doy in validDOYFF28]) 
                # (*) convert from DOY to month/day
                ls28 = dt.datetime.strptime(str(year) + "-" + str(doyLS28), "%Y-%j").strftime("%m/%d")
                ls32 = dt.datetime.strptime(str(year) + "-" + str(doyLS32), "%Y-%j").strftime("%m/%d")
                ff32 = dt.datetime.strptime(str(year) + "-" + str(doyFF32), "%Y-%j").strftime("%m/%d")
                ff28 = dt.datetime.strptime(str(year) + "-" + str(doyFF28), "%Y-%j").strftime("%m/%d")
                fdDict[fdDictKey] = np.array([year, station, ls28, ls32, ff32, ff28], dtype="object")

        # return df
        return pd.DataFrame.from_dict(data=fdDict, orient="index", columns=["YEAR", "STATION", "LAST SPR 28", "LAST SPR 32", "FIRST FALL 32", "FIRST FALL 28"])


    # formatting fd data according to the StateCU historic input
    fdDataFormatLine = ""
    with open(origFdFP, "r") as hfdFile:
        fdLines = hfdFile.readlines()
    # #>EndHeader line tells us how StateCU will read in the data
    for l, line in enumerate(fdLines):
        if "#>Temperatures are degrees F" in line:
            fdBreakLine = fdLines[l+1]
            fdColumnHeaderLine1 = fdLines[l+2]
            fdColumnHeaderLine2 = fdLines[l+3]
            fdDataFormatLine = fdLines[l+4]
            fdUnitsLine = fdLines[l+6]
            break
    fdUnitsLine = fdUnitsLine.replace("1950", str(min(sorted(set(syntheticPTDF["YEAR"])))))
    fdUnitsLine = fdUnitsLine.replace("2013", str(max(sorted(set(syntheticPTDF["YEAR"])))))
    # make a list of the b/e indices
    fdBs, fdEs = [0], [fdDataFormatLine.find("e")]
    fdB, fdE = fdBs[0], fdEs[0]
    while fdB >= 0 and fdE >= 0:
        fdB, fdE = fdDataFormatLine.find("b", fdE + 1), fdDataFormatLine.find("e", fdE + 1)
        fdBs.append(fdB), fdEs.append(fdE)
    fdBs, fdEs = fdBs[:-1], fdEs[:-1]

    # helper function for spacing out the data appropriately when writing
    def DataFormatter(val, b, e, frontPad=True):
        numMaxChars = e - b
        numDataChars = len(val)
        formattedData = ""
        if frontPad:
            for _ in range(numMaxChars - numDataChars + 1):
                formattedData += " "
            formattedData += val
        else:
            formattedData += val
            for _ in range(numMaxChars - numDataChars):
                formattedData += " "
        return formattedData

    # generate the frost dates using a k-NN disaggregation analogue
    synthData = syntheticPTDF.copy()
    dailyDict = UpdateDailyT(synthData, dailyTDict)
    frostDatesDF = kNNFrostDateDisaggregation(synthData, monthlyDF, dailyDict)
    with open(outputFdFP, "w") as fdFile:
        fdFile.write(fdBreakLine)
        fdFile.write(fdColumnHeaderLine1)
        fdFile.write(fdColumnHeaderLine2)
        fdFile.write(fdDataFormatLine)
        fdFile.write(fdUnitsLine)

        # start reading in the data in the appropriate order
        for yr in sorted(set(synthData["YEAR"])):
            yrIdx = frostDatesDF["YEAR"] == yr
            for stn in sorted(set(synthData["STATION"])):
                stnIdx = frostDatesDF["STATION"] == stn
                fdLine = ""
                fdLine += DataFormatter(str(yr), fdBs[0], fdEs[0]) + " "
                fdLine += DataFormatter(stationDict[stn], fdBs[1], fdEs[1], frontPad=False) + " "
                for cn, col in enumerate(["LAST SPR 28", "LAST SPR 32", "FIRST FALL 32", "FIRST FALL 28"]):
                    fdVal = frostDatesDF.loc[yrIdx & stnIdx, col].values[0]
                    fdLine += DataFormatter(fdVal, fdBs[2 + cn], fdEs[2 + cn])
                fdLine += "\n"
                fdFile.write(fdLine)


# only execute synthesizing of precip/temp if this script is actually executing, not when it's loading as a module
if __name__ == "__main__":
    # environment arguments
    dataRepo = sys.argv[1]
    repoName = dataRepo.replace("/", "_")
    scn = int(sys.argv[2])
    sim = int(sys.argv[3])
    
    # load in data
    rngs = np.load(syntheticDir + r"/{}/Scenario{}/{}_Scenario{}_SynthPT_RNGs.npy".format(dataRepo, scn + 1, repoName, scn + 1), allow_pickle=True).tolist()
    monthlyDF = pd.read_csv(syntheticDir + r"/{}/Scenario{}/{}_Scenario{}_UCRBMonthly.csv".format(dataRepo, scn + 1, repoName, scn + 1))
    gmmhmmDict = np.load(syntheticDir + r"/{}/Scenario{}/{}_Scenario{}_MultisiteCGMCs.npy".format(dataRepo, scn + 1, repoName, scn + 1), allow_pickle=True).item()
    copulaeDict = np.load(syntheticDir + r"/{}/Scenario{}/{}_Scenario{}_Copulae.npy".format(dataRepo, scn + 1, repoName, scn + 1), allow_pickle=True).item()
    dailyTDict = np.load(syntheticDir + r"/{}/Scenario{}/{}_Scenario{}_DailyTs.npy".format(dataRepo, scn + 1, repoName, scn + 1), allow_pickle=True).item()

    # output filepaths for the .prc/.tem/.fd data
    outputPrcFP = syntheticDir + r"/{}/Scenario{}/{}_COclim_Sim{}.prc".format(dataRepo, scn + 1, repoName, sim + 1)
    outputTemFP = syntheticDir + r"/{}/Scenario{}/{}_COclim_Sim{}.tem".format(dataRepo, scn + 1, repoName, sim + 1)
    outputFdFP = syntheticDir + r"/{}/Scenario{}/{}_COclim_Sim{}.fd".format(dataRepo, scn + 1, repoName, sim + 1)

    # actually run the synthesizing functions
    # -- setting the random number generator to parallel-tolerant random state
    rng = rngs[sim]
    # -- synthesizing disaggregated precipitation
    fullSynthYears = [y for y in range(min(gmmhmmDict["precipDF"].index.values), max(gmmhmmDict["precipDF"].index.values)+1)]
    annualSample = gmmhmmDict["model"].sample(len(fullSynthYears))[0]
    monthlySample = kNNPrecipDisaggregation(annualSample, monthlyDF, gmmhmmDict["precipDF"], fullSynthYears)
    # -- synthesizing conditional temperature based on disaggregated precipitation
    syntheticPTDF = PTPairGenerator(monthlySample, gmmhmmDict, copulaeDict)
    # -- write the corresponding .prc/.tem/.fd files
    WritePrecipAndTemp()
    WriteFrostDate()
