# import
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import scipy.stats
from scipy.stats import rankdata, mannwhitneyu, levene, cramervonmises_2samp, ks_2samp, ecdf
from multiprocessing import Process


# environment variables
dataToProcess = sys.argv[1]
repoName = dataToProcess.replace("/", "_")
scenario = int(sys.argv[2])
numSims = int(sys.argv[3])


# filepaths
statecuDir = os.path.dirname(os.path.dirname(__file__))
processedDir = statecuDir + r"/processed/NOAA"
syntheticDir = statecuDir + r"/synthetic/{}/Scenario{}".format(dataToProcess, scenario + 1)
plotsDir = statecuDir + r"/plots"
origPrcFP = statecuDir + r"/cdss-dev/cm2015_StateCU/StateCU/COclim2015.prc"
origTemFP = statecuDir + r"/cdss-dev/cm2015_StateCU/StateCU/COclim2015.tem"
origFdFP = statecuDir + r"/cdss-dev/cm2015_StateCU/StateCU/COclim2015.fd"


# stationDict, its inverse, and month numbers
stationDict = {"Altenbern": "USC00050214", "Collbran": "USC00051741",
               "Eagle County": "USW00023063", "Fruita": "USC00053146",
               "Glenwood Springs": "USC00053359", "Grand Junction": "USC00053489",
               "Grand Lake": "USC00053500", "Green Mt Dam": "USC00053592",
               "Kremmling": "USC00054664", "Meredith": "USC00055507",
               "Rifle": "USC00057031", "Yampa": "USC00059265"}
iStationDict = {v: k for k, v in stationDict.items()}
monthNumDict = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}


# load in the p/T data from .prc/.tem files, create single dataframe
def LoadSyntheticPTData():
    # read in the original prc file
    dataFormatLine = ""
    with open(origPrcFP, "r") as histPrc:
        lines = histPrc.readlines()
        # #>-e/b line tells us how to read in the data, drop the rest that start with "#"
        for line in lines:
            if "#>-e-b" in line:
                dataFormatLine = line
                break

    # helper function to break down the data in the lines
    def ExtractCUData(dataFormat, l):
        bIdx, j, data = 0, 0, []
        eIdx = dataFormat.find("e")
        while bIdx >= 0 and eIdx >= 0:
            if j == 0:
                year = int(l[bIdx:eIdx + 1])
            if j == 1:
                stationID = "".join([s for s in l[bIdx:eIdx + 1] if s != " "])
                station = iStationDict[stationID]
            if j > 1:
                datum = float("".join([s for s in l[bIdx:eIdx + 1] if s != " "]))
                data.append(datum)
            bIdx = dataFormat.find("b", eIdx + 1)
            eIdx = dataFormat.find("e", eIdx + 1)
            j += 1

        # return
        return year, station, data[:-1]

    # get our .prc, .tem files we have in the synthetic directory
    nPrcFiles = len([f for f in os.listdir(syntheticDir) if ".prc" in f])
    nTemFiles = len([f for f in os.listdir(syntheticDir) if ".tem" in f])
    
    # organize so that the 1st element is the 1st realization, etc
    simFileStr = "{}_COclim_Sim".format(repoName) 
    prcFiles = [simFileStr + "{}.prc".format(n) for n in range(1, nPrcFiles+1)] 
    temFiles = [simFileStr + "{}.tem".format(n) for n in range(1, nTemFiles+1)]

    # construct a dictionary for all the samples
    synthDF = pd.DataFrame(columns=["STATION", "YEAR", "MONTH", "PRCP", "TAVG"])
    synthDict = {n: synthDF.copy() for n in range(numSims)}
    # read in the synthetic precipitation, building each simulation's dictionary
    for n in range(numSims):
        # reading in the precipitation data
        with open(syntheticDir + "/" + prcFiles[n], "r") as synthPrc:
            lines = synthPrc.readlines()
            simPrcDict, simPrcKey = {}, 0
            for line in lines:
                if "#" not in line and "CYR" not in line:
                    extractedData = ExtractCUData(dataFormatLine, line)
                    for m, month in enumerate(list(monthNumDict.keys())):
                        simPrcKey += 1
                        simPrcDict[simPrcKey] = [extractedData[0], extractedData[1], month, extractedData[2][m]]
            synthSimDF = pd.DataFrame.from_dict(simPrcDict, orient="index", columns=["YEAR", "STATION", "MONTH", "PRCP"])
            synthSimDF = synthSimDF[synthSimDF["STATION"].notna()]

        # reading in the temperature data
        with open(syntheticDir + "/" + temFiles[n], "r") as synthTem:
            lines = synthTem.readlines()
            simTemDict, simTemKey = {}, 0
            for line in lines:
                if "#" not in line and "CYR" not in line:
                    extractedData = ExtractCUData(dataFormatLine, line)
                    for m, month in enumerate(list(monthNumDict.keys())):
                        simTemKey += 1
                        simTemDict[simTemKey] = [extractedData[0], extractedData[1], month, extractedData[2][m]]
            synthSimDF = pd.concat([synthSimDF, pd.DataFrame.from_dict(simTemDict, orient="index", columns=["YEAR", "STATION", "MONTH", "TAVG"])], ignore_index=True)
            synthSimDF = synthSimDF[synthSimDF["STATION"].notna()]

        # condensing dataframe, converting in->m and degF->degC, inserting in the large dictionary
        fullSynthSimDF = synthSimDF.copy()
        fullSynthSimDF.loc[fullSynthSimDF["PRCP"].notna().values, "TAVG"] = fullSynthSimDF.loc[fullSynthSimDF["TAVG"].notna(), "TAVG"].values
        fullSynthSimDF = fullSynthSimDF.dropna()
        fullSynthSimDF["PRCP"] = fullSynthSimDF["PRCP"].apply(lambda x: x*0.0254)
        fullSynthSimDF["TAVG"] = fullSynthSimDF["TAVG"].apply(lambda x: (x-32.)*(5./9))
        synthDict[n] = fullSynthSimDF

    # return the dictionary
    return synthDict


# direct histogram between historic/cmip6 and synthetic
def GMMHMMHistograms():
    HistComparePlot, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all", sharey="all")
    HistComparePlot.suptitle("Comparison of NOAAObs and Scenario Sample Data via Histogram".format(repoName))
    HistComparePlot.supxlabel("Total Annual Precipitation [m]"), HistComparePlot.supylabel("Probability Density [-]")
    for i, axis in enumerate(axes.flat):
        # plot things to the axis
        axis.grid()
        axis.set(title=stations[i])
        axis.set(xscale="log")
        logbins = np.logspace(min(HMMDict["precipDF"][stations[i]].values), max(HMMDict["precipDF"][stations[i]].values), 10)
        # plot histogram of samples
        annualSamples = np.array([])
        for n in synthPTDict.keys():
            stationIdx = synthPTDict[n]["STATION"] == stations[i]
            for year in sorted(set(synthPTDict[n]["YEAR"])):
                yearIdx = synthPTDict[n]["YEAR"] == year
                annualSample = np.nansum(synthPTDict[n].loc[stationIdx & yearIdx, "PRCP"].values)
                annualSamples = np.append(annualSamples, annualSample)
        axis.hist(annualSamples, density=True, bins=logbins, color="grey", alpha=1)

        # plot the histogram of historic
        axis.hist(10 ** HMMDict["precipDF"][stations[i]].values, density=True, bins=logbins, color="black", histtype="step")
    plt.tight_layout()
    HistComparePlot.savefig(plotsDir + r"/swg/{}/Scenario{}/{}_Scenario{}_HistogramComparison.svg".format(dataToProcess, scenario + 1, repoName, scenario + 1))
    plt.close()


# plot spatial correlation matrices for the sample precip generated by the GMMHMM fit
def SynthSpatialCorr():
    # the segmented colormap for each of the spatial correlations
    #cmapCorr = plt.get_cmap("gnuplot", 11)
    cmapCorr = plt.get_cmap("gnuplot")
    
    # plotting
    for wvar in ["PRCP", "TAVG"]:
        # all months for all years per each station
        sampleCorrDF = pd.DataFrame(columns=stations, dtype=float)
        for station in stations:
            fullStationSample = np.array([])
            for n in synthPTDict.keys():
                stationIdx = synthPTDict[n]["STATION"] == station
                for year in sorted(set(synthPTDict[n]["YEAR"])):
                    yearIdx = synthPTDict[n]["YEAR"] == year
                    fullStationSample = np.append(fullStationSample, synthPTDict[n].loc[stationIdx & yearIdx, wvar].values)
            sampleCorrDF[station] = fullStationSample
        actualSampleCorrelation = sampleCorrDF.corr(method="pearson")
        SampleCorrPlot, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), sharex="all", sharey="all")
        sampCorrColors = axes.matshow(actualSampleCorrelation, vmin=0, vmax=1, cmap=cmapCorr)
        # for i, station1 in enumerate(stations):
        #     for j, station2 in enumerate(stations):
        #         plt.text(j, i, '{:.2f}'.format(actualSampleCorrelation.at[station1, station2]), ha='center', va='center', backgroundcolor=[0, 0, 0, 0.25], color="white")
        SampleCorrPlot.colorbar(sampCorrColors, label="Scenario {} Sample Monthly Spatial Correlation [Pearson, -]".format(wvar))
        plt.xticks(range(len(stations)), stations, rotation=75)
        plt.yticks(range(len(stations)), stations, rotation=0)
        SampleCorrPlot.savefig(plotsDir + r"/swg/{}/Scenario{}/{}_Scenario{}_{}SampleSpatialCorrelation.svg".format(dataToProcess, scenario + 1, repoName, scenario + 1, wvar))
        plt.close()
    
        # all years per each station per each month
        for month in months:
            monthlyCorrDF = pd.DataFrame(columns=stations, dtype=float)
            for station in stations:
                monthStationSample = np.array([])
                for n in synthPTDict.keys():
                    monthIdx = synthPTDict[n]["MONTH"] == month
                    stationIdx = synthPTDict[n]["STATION"] == station
                    for year in sorted(set(synthPTDict[n]["YEAR"])):
                        yearIdx = synthPTDict[n]["YEAR"] == year
                        monthStationSample = np.append(monthStationSample, synthPTDict[n].loc[stationIdx & yearIdx & monthIdx, wvar].values)
                monthlyCorrDF[station] = monthStationSample
            monthSampleCorrelation = monthlyCorrDF.corr(method="pearson")
            MonthlyCorrPlot, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), sharex="all", sharey="all")
            monthlyCorrColors = axes.matshow(monthSampleCorrelation, vmin=0, vmax=1, cmap=cmapCorr)
            # for i, station1 in enumerate(stations):
            #     for j, station2 in enumerate(stations):
            #         plt.text(j, i, '{:.2f}'.format(monthSampleCorrelation.at[station1, station2]), ha='center', va='center', backgroundcolor=[0, 0, 0, 0.25], color="white")
            MonthlyCorrPlot.colorbar(monthlyCorrColors, label="Scenario {} Sample {} Spatial Correlation [Pearson, -]".format(wvar, month))
            plt.xticks(range(len(stations)), stations, rotation=75)
            plt.yticks(range(len(stations)), stations, rotation=0)
            MonthlyCorrPlot.savefig(plotsDir + r"/swg/{}/Scenario{}/{}_Scenario{}_{}SampleSpatialCorrelation_{}.svg".format(dataToProcess, scenario + 1, repoName, scenario + 1, wvar, month))
            plt.close()


# plot the "flow duration curve" analogue for precipitation (cumulative frequency of each trace of yearly precip by station)
def GMMHMMCumFreq():
    precipCumFreqPlot, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all", sharey="all")
    precipCumFreqPlot.suptitle("{} {} Cumulative Frequency Curves for Monthly Precipitation".format(len(synthPTDict.keys()), repoName))
    precipCumFreqPlot.supxlabel("% Exceedance"), precipCumFreqPlot.supylabel("Total Monthly Precipitation [m]")
    for i, axis in enumerate(axes.flat):
        # pre-processing
        axis.grid()
        axis.set(title=stations[i])
        axis.set(yscale="log", ylim=[2.54E-4, 0.5])

        # working out the percent exceedance for each sample
        for n in range(len(synthPTDict.keys())):
            # grab individual trace, put in descending order
            trace = np.array([])
            stationIdx = synthPTDict[n]["STATION"] == stations[i]
            for year in sorted(set(synthPTDict[n]["YEAR"])):
                yearIdx = synthPTDict[n]["YEAR"] == year
                trace = np.append(trace, synthPTDict[n].loc[stationIdx & yearIdx, "PRCP"].values)
            descendingTrace = sorted(trace, reverse=True)
            # check how probable it is to have the precipitation meet or exceed this value
            exceedance = np.array([np.nansum(descendingTrace >= point) / len(descendingTrace) for point in descendingTrace])
            axis.plot(100 * exceedance, descendingTrace, c="grey", alpha=0.33, label="synthetic", rasterized=True)
        # adding in the historic exceedance
        historic = HMMDict["monthlyDF"][stations[i]].values
        descendingHistoric = sorted(historic, reverse=True)
        histExceedance = np.array([np.nansum(descendingHistoric >= point) / len(descendingHistoric) for point in descendingHistoric])
        axis.plot(100 * histExceedance, descendingHistoric, c="black", label="historic")
    plt.tight_layout()
    precipCumFreqPlot.savefig(plotsDir + r"/swg/{}/Scenario{}/{}_Scenario{}_SampleCumFreq.svg".format(dataToProcess, scenario + 1, repoName, scenario + 1))
    plt.close()


# check that the generated monthly samples have a Markovian structure
def GMMHMMSynthMarkovian():
    numLags = 24
    for d, station in enumerate(stations):
        SamplesTemporalityPlot, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 9), sharex="all")
        SamplesTemporalityPlot.suptitle("{} Markovian Process Check for Monthly Scenario Samples @ {}".format(repoName, station))
        SamplesTemporalityPlot.supxlabel("Lag")

        for i, axis in enumerate(axes.flat):
            # pre-plot stuff
            axis.grid()
            # ACF or PACF
            if i == 0:
                acfLagDict = {lag: [] for lag in range(numLags + 1)}
                for n in range(len(synthPTDict.keys())):
                    monthlyData4ACF = np.array([])
                    stationIdx = synthPTDict[n]["STATION"] == station
                    for year in sorted(set(synthPTDict[n]["YEAR"])):
                        yearIdx = synthPTDict[n]["YEAR"] == year
                        monthlyData4ACF = np.append(monthlyData4ACF, synthPTDict[n].loc[stationIdx & yearIdx, "PRCP"].values)
                    sequenceACF = acf(monthlyData4ACF, nlags=numLags)
                    for lag in range(numLags + 1):
                        acfLagDict[lag].append(sequenceACF[lag])
                axis.boxplot(acfLagDict.values(), sym="", positions=np.arange(len(acfLagDict.keys())))
                axis.stem(0, 1, linefmt="black")
                plot_acf(ax=axis, x=monthlyData4ACF, vlines_kwargs={"color": "None"}, color="None")
                axis.set(ylabel="ACF [-]", xlim=(-0.25, numLags + 0.25))
                axis.hlines(0, xmin=0, xmax=numLags, color="red")
            if i == 1:
                pacfLagDict = {lag: [] for lag in range(numLags + 1)}
                for n in range(len(synthPTDict.keys())):
                    monthlyData4PACF = np.array([])
                    stationIdx = synthPTDict[n]["STATION"] == station
                    for year in sorted(set(synthPTDict[n]["YEAR"])):
                        yearIdx = synthPTDict[n]["YEAR"] == year
                        monthlyData4PACF = np.append(monthlyData4PACF, synthPTDict[n].loc[stationIdx & yearIdx, "PRCP"].values)
                    sequencePACF = pacf(monthlyData4PACF, nlags=numLags)
                    for lag in range(numLags + 1):
                        pacfLagDict[lag].append(sequencePACF[lag])
                axis.boxplot(pacfLagDict.values(), sym="", positions=np.arange(len(pacfLagDict.keys())))
                axis.stem(0, 1, linefmt="black")
                plot_pacf(ax=axis, x=monthlyData4PACF, method="ywm", vlines_kwargs={"color": "None"}, color="None")
                axis.set(ylabel="PACF [-]", xlim=(-0.25, numLags + 0.25))
                axis.hlines(0, xmin=0, xmax=numLags, color="red")
        plt.tight_layout()
        SamplesTemporalityPlot.savefig(plotsDir + r"/swg/{}/Scenario{}/{}_Scenario{}_GMMHMMFit_SampleMarkovianStructure_{}.svg".format(dataToProcess, scenario+1, repoName, scenario+1, station.replace(" ", "")))
        plt.close()


# plot monthly statistics for each station
def GMMHMMMonthlyMoments():
    for s, station in enumerate(stations):
        # reshape the disaggregated data to perform statistics: (data points across all sample years) x (months) 
        histMonthly = HMMDict["monthlyDF"][station].astype(float).values
        fullSynthMonthly = np.full(shape=(len(set(synthPTDict[0]["YEAR"])) * len(synthPTDict.keys()), len(months)), fill_value=np.NaN)
        for n in range(len(synthPTDict.keys())):
            stationIdx = synthPTDict[n]["STATION"] == station
            for y, year in enumerate(sorted(set(synthPTDict[n]["YEAR"]))):
                yearIdx = synthPTDict[n]["YEAR"] == year
                fullSynthMonthly[n * len(set(synthPTDict[n]["YEAR"])) + y, :] = synthPTDict[n].loc[stationIdx & yearIdx, "PRCP"].values
        synthMonthly = fullSynthMonthly.copy()
        
        monthlyMomentsPlot, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 9), sharex="all")
        monthlyMomentsPlot.suptitle("{} Null-Hypothesis Plots @ {}".format(repoName, station))
        monthlyMomentsPlot.supxlabel("Months")
        for i, axis in enumerate(axes.flat):
            # pre-plotting
            axis.grid()

            # for the month we're interested in
            histDict = {month: np.array([]) for month in months}
            synthDict = {month: np.array([]) for month in months}
            for m, month in enumerate(months):
                histDict[month] = histMonthly[m::len(months)]
                synthDict[month] = synthMonthly[:, m]

            boxWidth = 0.1
            # subplots: monthly precipitation
            if i == 0:
                axis.set(ylabel="Monthly Precipitation [m]")
                axis.boxplot(histDict.values(), sym=None, positions=np.arange(len(months)) - boxWidth,
                        widths=boxWidth, patch_artist=True, boxprops={"facecolor": "royalblue"}, flierprops={"marker": ".", "markeredgecolor": "royalblue"})
                axis.boxplot(synthDict.values(), sym=None, positions=np.arange(len(months)) + boxWidth,
                        widths=boxWidth, patch_artist=True, boxprops={"facecolor": "lightgrey"}, flierprops={"marker": ".", "markeredgecolor": "lightgrey"})
                axis.set_xticks(np.arange(len(months)))
                axis.set_xticklabels(months)
                axis.tick_params(axis="x", labelsize=len(months))

            # subplots: Mann-Whitney U p value (tests median)
            if i == 1:
                axis.set(ylabel="Mann-Whitney U p-Value [-]", ylim=[0, 1])
                ranksumPValues = [stats.mannwhitneyu(histDict[month], synthDict[month])[1] for month in months]
                axis.bar([x for x in range(len(months))], ranksumPValues, color="darkgrey", zorder=10)
                axis.hlines(0.05, xmin=-0.5, xmax=len(months) - 0.5, colors="black", linestyles="dashed", zorder=11)

            # subplots: Levene p value (tests std)
            if i == 2:
                axis.set(ylabel="Levene p-Value [-]", ylim=[0, 1])
                levenePValues = [stats.levene(histDict[month], synthDict[month])[1] for month in months]
                axis.bar([x for x in range(len(months))], levenePValues, color="darkgrey", zorder=10)
                axis.hlines(0.05, xmin=-0.5, xmax=len(months) - 0.5, colors="black", linestyles="dashed", zorder=11)

        # post-plotting
        plt.tight_layout()
        monthlyMomentsPlot.savefig(plotsDir + r"/swg/{}/Scenario{}/{}_Scenario{}_MonthlyMoments_{}.svg".format(dataToProcess, scenario + 1, repoName, scenario + 1, station.replace(" ", "")))
        plt.close()


# compare the historic and synthetic Kendall/Spearman correlation coefficients
def PTPairCorrelations(): 
    corrFig, axis = plt.subplots(nrows=1, ncols=1, figsize=(14, 9))
    corrFig.suptitle("{} Correlation of P/T Data by Month".format(repoName))
    corrFig.supxlabel("Month"), corrFig.supylabel("Correlation Coefficient [-]")
    # creating the data
    histDict = {c: {month: np.NaN for month in months} for c in ["Kendall", "Spearman"]}
    synthDict = {c: {month: [] for month in months} for c in ["Kendall", "Spearman"]}
    for month in months:
        histPrcp, histTavg = CopulaDict[month]["PRCP"], CopulaDict[month]["TAVG"]
        hptDF = pd.DataFrame({"P": histPrcp, "T": histTavg})
        histDict["Kendall"][month] = hptDF.corr(method="kendall")["P"]["T"]
        histDict["Spearman"][month] = hptDF.corr(method="spearman")["P"]["T"]
        for n in range(nSamples):
            monthIdx = synthPTDict[n]["MONTH"] == month
            pVals, tVals = [], []
            for year in sorted(set(synthPTDict[n]["YEAR"])):
                yearIdx = synthPTDict[n]["YEAR"] == year
                pVals.append(np.nanmean(synthPTDict[n].loc[monthIdx & yearIdx, "PRCP"].values))
                tVals.append(np.nanmean(synthPTDict[n].loc[monthIdx & yearIdx, "TAVG"].values))
            sptDF = pd.DataFrame({"P": pVals, "T": tVals})
            synthKendallCorr = sptDF.corr(method="kendall")["P"]["T"]
            synthSpearmanCorr = sptDF.corr(method="spearman")["P"]["T"]
            synthDict["Kendall"][month].append(synthKendallCorr)
            synthDict["Spearman"][month].append(synthSpearmanCorr) 
    
    # plotting
    synthColor, boxWidth = "grey", 0.1
    axis.grid()
    axis.set_ylim(-1, 1)
    axis.hlines(0, xmin=0, xmax=11, colors="black", linestyles="dashed")
    #axis.boxplot(synthDict["Kendall"].values(), sym="", positions=range(len(months)), widths=boxWidth, patch_artist=True, boxprops={"facecolor": synthColor})
    #axis.boxplot(synthDict["Spearman"].values(), sym="", positions=range(len(months)), widths=boxWidth, patch_artist=True, notch=True, boxprops={"facecolor": synthColor})
    axis.plot(range(len(months)), histDict["Kendall"].values(), color="black", marker="s", label=r"Kendall $\tau$")
    axis.plot(range(len(months)), histDict["Spearman"].values(), color="black", marker="d", label=r"Spearman $\rho$")
    k, s = [], []
    for month in months:
        k.append(np.nanmean(synthDict["Kendall"][month]))
        s.append(np.nanmean(synthDict["Spearman"][month]))
    axis.scatter(range(len(months)), k, color=synthColor, marker="s", zorder=15)
    axis.scatter(range(len(months)), s, color=synthColor, marker="d", zorder=15) 
    axis.legend()
    axis.set_xticks(range(len(months)))
    axis.set_xticklabels(months, rotation=45)
    plt.tight_layout()
    corrFig.savefig(plotsDir + r"/swg/{}/Scenario{}/{}_Scenario{}_SyntheticKendallSpearman.svg".format(dataToProcess, scenario + 1, repoName, scenario + 1))
    plt.close()


# scatterplot and histogram for the historic and synthetic
def PTPairScatterplots():
    fitYears = HMMDict["precipDF"].index.values
    histYearsIdx = (MonthlyDF["YEAR"] >= min(fitYears)) & (MonthlyDF["YEAR"] <= max(fitYears))
    for station in stations:
        histStationIdx = MonthlyDF["NAME"] == station
        histsynthDistFig = plt.figure(figsize=(14, 9))
        subFigs = histsynthDistFig.subfigures(3, 4)
        for i, subFig in enumerate(subFigs.flat):
            histMonthIdx = MonthlyDF["MONTH"] == months[i]
            axes = subFig.subplots(2, 2, gridspec_kw={"width_ratios": [4, 1], "height_ratios": [1, 3]})
            subFig.subplots_adjust(wspace=0, hspace=0)
            # match CopulaDict[station][months[i]]["BestCopula"][1]:
            #     case "Independence":
            #         synthColor = "grey"
            #     case "Frank":
            #         synthColor = "royalblue"
            #     case "Gaussian":
            #         synthColor = "red"
            synthColor = "grey"
            for j, axis in enumerate(axes.flat):
                if j == 0:
                    # precip histogram comparison
                    synthPrcp = np.array([])
                    for n in range(nSamples):
                        synthPrcp = np.append(synthPrcp, synthPTDict[n].loc[(synthPTDict[n]["STATION"] == station) & (synthPTDict[n]["MONTH"] == months[i]), "PRCP"])
                    axis.hist(synthPrcp[np.isfinite(synthPrcp.astype(float))], density=True, color=synthColor)
                    axis.hist(MonthlyDF.loc[histStationIdx & histMonthIdx & histYearsIdx, "PRCP"].values, density=True, color="black", histtype="step")
                    axis.set(xticks=[], yticks=[])
                if j == 1:
                    # label for the month
                    axis.axis("off")
                    axis.text(0.5, 0.5, months[i], transform=axis.transAxes, va="center", ha="center")
                if j == 2:
                    # scatterplot
                    for n in range(nSamples):
                        stationIdx = synthPTDict[n]["STATION"] == station
                        monthIdx = synthPTDict[n]["MONTH"] == months[i]
                        axis.scatter(synthPTDict[n].loc[stationIdx & monthIdx, "PRCP"],
                                     synthPTDict[n].loc[stationIdx & monthIdx, "TAVG"],
                                     marker=".", facecolors=synthColor, alpha=0.1, rasterized=True)
                    axis.scatter(MonthlyDF.loc[histStationIdx & histMonthIdx & histYearsIdx, "PRCP"].values, 
                                 MonthlyDF.loc[histStationIdx & histMonthIdx & histYearsIdx, "TAVG"].values, marker="o", facecolors="none", edgecolors="black")
                if j == 3:
                    # temp histogram comparison
                    synthTavg = np.array([])
                    for n in range(nSamples):
                        synthTavg = np.append(synthTavg, synthPTDict[n].loc[(synthPTDict[n]["STATION"] == station) & (synthPTDict[n]["MONTH"] == months[i]), "TAVG"])
                    axis.hist(synthTavg[np.isfinite(synthTavg.astype(float))], density=True, color=synthColor, orientation="horizontal")
                    axis.hist(MonthlyDF.loc[histStationIdx & histMonthIdx & histYearsIdx, "TAVG"].values, density=True, color="black", histtype="step", orientation="horizontal")
                    axis.set(xticks=[], yticks=[])
        histsynthDistFig.savefig(plotsDir + r"/swg/{}/Scenario{}/{}_Scenario{}_SyntheticDistributionComparison_{}.svg".format(dataToProcess, scenario+1, repoName, scenario+1, station.replace(" ", "")))
        plt.close()


# statistics for the historic/synthetic precipitation/temperature distributions
# -- Mann-Whitney U (MWU), median
# -- Levene, std
# -- Kolmogorov-Smirnov (T_n), empirical distribution sensitive to mean
def PTPairNullHypothesis():
    goodnessStats, barWidth = ["MWU", "Levene", "$T_n$"], 0.5
    fitYears = HMMDict["precipDF"].index.values
    histYearsIdx = (MonthlyDF["YEAR"] >= min(fitYears)) & (MonthlyDF["YEAR"] <= max(fitYears))
    for station in stations:
        histStationIdx = MonthlyDF["NAME"] == station
        StatisticsPlots, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all", sharey="all")
        StatisticsPlots.suptitle("Null-Hypothesis Statistics @ {}".format(station))
        StatisticsPlots.supylabel("p-Value [-]")
        for i, axis in enumerate(axes.flat):
            histMonthIdx = MonthlyDF["MONTH"] == months[i]
            histPrcp = MonthlyDF.loc[histStationIdx & histMonthIdx & histYearsIdx, "PRCP"].astype(float).values 
            histTavg = MonthlyDF.loc[histStationIdx & histMonthIdx & histYearsIdx, "TAVG"].astype(float).values
            synthPrcp, synthTavg = np.array([], dtype=float), np.array([], dtype=float)
            for n in range(nSamples):
                stationIdx = synthPTDict[n]["STATION"] == station
                monthIdx = synthPTDict[n]["MONTH"] == months[i]
                synthPrcp = np.append(synthPrcp, synthPTDict[n].loc[stationIdx & monthIdx, "PRCP"].astype(float).values)
                synthTavg = np.append(synthTavg, synthPTDict[n].loc[stationIdx & monthIdx, "TAVG"].astype(float).values)
            # acutal metrics
            prcpMWU, tavgMWU = mannwhitneyu(histPrcp, synthPrcp, nan_policy="omit"), mannwhitneyu(histTavg, synthTavg, nan_policy="omit")
            prcpLevene, tavgLevene = levene(histPrcp, synthPrcp[np.isfinite(synthPrcp)]), levene(histTavg, synthTavg[np.isfinite(synthTavg)])
            prcpKS, tavgKS = ks_2samp(histPrcp, synthPrcp[np.isfinite(synthPrcp)]), ks_2samp(histTavg, synthTavg[np.isfinite(synthTavg)])

            axis.grid()
            axis.set(title=months[i], xlim=[-barWidth * 1.5, 2 * (len(goodnessStats) - 1 + barWidth)], ylim=[0, 1])
            axis.bar([2 * x - barWidth / 2 for x in range(len(goodnessStats))], [prcpMWU[1], prcpLevene[1], prcpKS.pvalue], width=barWidth, color="royalblue", zorder=10, label="PRCP")
            axis.bar([2 * x + barWidth / 2 for x in range(len(goodnessStats))], [tavgMWU[1], tavgLevene[1], tavgKS.pvalue], width=barWidth, color="orangered", zorder=10, label="TAVG")
            if i == 11:
                axis.legend().set_zorder(11)
            axis.hlines(0.05, -2 * barWidth, 2 * (len(goodnessStats) + barWidth), color="black", linestyles="dashed", zorder=11)
            axis.set_xticks([2 * x for x in range(len(goodnessStats))])
            axis.set_xticklabels(labels=goodnessStats)
        plt.tight_layout()
        StatisticsPlots.savefig(plotsDir + r"/swg/{}/Scenario{}/{}_Scenario{}_SyntheticGoodnessMetrics_{}.svg".format(dataToProcess, scenario + 1, repoName, scenario + 1, station.replace(" ", "")))
        plt.close()


# confirm that the synthetic frost dates aren't out to lunch from the historic dataset
def FDPlots():
    # read in the historic frost dates
    with open(origFdFP, "r") as histFD:
        lines = histFD.readlines()

    # #>-e/b line tells us how to read in the data, drop the rest that start with "#"
    dataFormatLine = ""
    for line in lines:
        if "#>-e-b" in line:
            dataFormatLine = line
            break

    # helper function to break down the data in the lines
    def ExtractCUData(dataFormat, l):
        bIdx, j, doys = 0, 0, []
        eIdx = dataFormat.find("e")
        while bIdx >= 0 and eIdx >= 0:
            if j == 0:
                year = int(l[bIdx:eIdx + 1])
            if j == 1:
                station = np.NaN
                stationID = "".join([s for s in l[bIdx:eIdx + 1] if s != " "])
                if stationID in stationDict.values():
                    station = [list(stationDict.keys())[n] for n, val in enumerate(stationDict.values()) if val == stationID][0]
            if j > 1:
                mmdd = "".join([s for s in l[bIdx:eIdx + 1] if s != " "])
                doy = np.NaN
                if mmdd != "-999.0":
                    doy = pd.Period(str(year) + "-" + mmdd[:2] + "-" + mmdd[3:], freq="D").day_of_year
                doys.append(doy)
            bIdx = dataFormat.find("b", eIdx + 1)
            eIdx = dataFormat.find("e", eIdx + 1)
            j += 1

        # return
        return year, station, *doys

    # read in the historical data and convert MM/DD to DOY
    histDF = pd.DataFrame(columns=["YEAR", "STATION", "LAST SPR 28", "LAST SPR 32", "FIRST FALL 32", "FIRST FALL 28"])
    for line in lines:
        if "#" not in line and "CYR" not in line:
            histDF.loc[len(histDF)] = ExtractCUData(dataFormatLine, line)
    histDF = histDF[histDF["STATION"].notna()]

    # read in the synthetic data and convert MM/DD to DOY
    synthDict, synthDictKey = {}, -1
    for f in os.listdir(syntheticDir):
        if f.endswith(".fd"):
            with open(syntheticDir + "/" + f, "r") as synthFD:
                lines = synthFD.readlines()
                for line in lines:
                    synthDictKey += 1
                    if "#" not in line and "CYR" not in line:
                        synthDict[synthDictKey] = ExtractCUData(dataFormatLine, line)
    synthDF = pd.DataFrame.from_dict(synthDict, orient="index", columns=["YEAR", "STATION", "LAST SPR 28", "LAST SPR 32", "FIRST FALL 32", "FIRST FALL 28"])
    synthDF = synthDF[synthDF["STATION"].notna()]

    # actually plot FD DOY histogram
    frostdateDOYHistogramFig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all")
    frostdateDOYHistogramFig.suptitle("{} Frost Date DOY Histograms".format(repoName))
    frostdateDOYHistogramFig.supxlabel("DOY"), frostdateDOYHistogramFig.supylabel("Probability [-]")
    for i, axis in enumerate(axes.flat):
        # plot info
        axis.grid()
        axis.set_title(list(stationDict.keys())[i])

        # historical data
        axis.hist(histDF.loc[histDF["STATION"] == list(stationDict.keys())[i], "LAST SPR 28"], density=True, color="black", histtype="step", linestyle="dashed")
        axis.hist(histDF.loc[histDF["STATION"] == list(stationDict.keys())[i], "LAST SPR 32"], density=True, color="black", histtype="step")
        axis.hist(histDF.loc[histDF["STATION"] == list(stationDict.keys())[i], "FIRST FALL 32"], density=True, color="black", histtype="step")
        axis.hist(histDF.loc[histDF["STATION"] == list(stationDict.keys())[i], "FIRST FALL 28"], density=True, color="black", histtype="step", linestyle="dashed")

        # synthetic data
        axis.hist(synthDF.loc[synthDF["STATION"] == list(stationDict.keys())[i], "LAST SPR 28"], density=True, color="darkgrey", linestyle="dashed")
        axis.hist(synthDF.loc[synthDF["STATION"] == list(stationDict.keys())[i], "FIRST FALL 28"], density=True, color="darkgrey", linestyle="dashed")
        axis.hist(synthDF.loc[synthDF["STATION"] == list(stationDict.keys())[i], "LAST SPR 32"], density=True, color="grey")
        axis.hist(synthDF.loc[synthDF["STATION"] == list(stationDict.keys())[i], "FIRST FALL 32"], density=True, color="grey")
    plt.tight_layout()
    frostdateDOYHistogramFig.savefig(plotsDir + r"/swg/{}/Scenario{}/{}_Scenario{}_FrostDateDOYHistograms.svg".format(dataToProcess, scenario+1, repoName, scenario+1))
    plt.close()

    # actually plot FD difference between LAST SPRING 28 and FIRST SPRING 28 
    for fltemp in [28, 32]:
        lastSpr = "LAST SPR 28" if fltemp == 28 else "LAST SPR 32"
        firstFall = "FIRST FALL 28" if fltemp == 28 else "FIRST FALL 32"
        frostdateDiffFig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all")
        frostdateDiffFig.suptitle("{} Frost Date {}Deg Histograms".format(repoName, fltemp))
        frostdateDiffFig.supxlabel("Days Between {}Deg".format(fltemp)), frostdateDiffFig.supylabel("Probability [-]")
        for i, axis in enumerate(axes.flat):
            # plot info
            axis.grid()
            axis.set_title(list(stationDict.keys())[i])

            # calculate the difference between historic last spring and first fall
            hls = histDF.loc[histDF["STATION"] == list(stationDict.keys())[i], lastSpr].values
            hff = histDF.loc[histDF["STATION"] == list(stationDict.keys())[i], firstFall].values
            axis.hist(hff - hls, density=True, color="black", histtype="step")
            
            # calculate the difference between synthetic last spring and first fall
            sls = synthDF.loc[synthDF["STATION"] == list(stationDict.keys())[i], lastSpr].values
            sff = synthDF.loc[synthDF["STATION"] == list(stationDict.keys())[i], firstFall].values
            axis.hist(sff - sls, density=True, color="grey")
        plt.tight_layout()
        frostdateDiffFig.savefig(plotsDir + r"/swg/{}/Scenario{}/{}_Scenario{}_FrostDate{}DiffHistograms.svg".format(dataToProcess, scenario+1, repoName, scenario+1, fltemp))
        plt.close()


# if we execute the script, load in the PT data and run the plotting on multiple threads
if __name__ == "__main__":
    # loading in the gmmhmm, copulae data
    MonthlyDF = pd.read_csv(processedDir + r"/NOAA_UCRBMonthly.csv")
    HMMDict = np.load(processedDir + r"/NOAA_MultisiteGMMHMMFit.npy", allow_pickle=True).item()
    CopulaDict = np.load(processedDir + r"/NOAA_CopulaFits.npy", allow_pickle=True).item()
    stations = HMMDict["precipDF"].columns.values
    months, nSamples = list(CopulaDict.keys()), numSims
    
    # loading synthetic data
    synthPTDict = LoadSyntheticPTData()

    # helper function for multithreading
    def MultiprocessSynthPTPlotting(fns, fninputs):
        pross = []
        for i, fn in enumerate(fns):
            p = Process(target=fn, args=fninputs[i])
            p.start()
            pross.append(p)
        for p in pross:
            p.join()
    plotFns = [GMMHMMHistograms, SynthSpatialCorr, GMMHMMCumFreq, GMMHMMSynthMarkovian, GMMHMMMonthlyMoments,
               PTPairCorrelations, PTPairScatterplots, PTPairNullHypothesis, FDPlots]
    MultiprocessSynthPTPlotting(plotFns, [()] * len(plotFns))

