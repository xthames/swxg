# imports
import os
import sys
import pandas as pd
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import statsmodels.api as sm
from scipy import stats
import warnings
from multiprocessing import Process
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.ar_model import AutoReg
from NOAAClimateDataReader import CalculateGeographicDistance


# environment variables
dataRepo = sys.argv[1]
repoName = dataRepo.replace("/", "_")
primaryStationDict = {"Altenbern": "USC00050214", "Collbran": "USC00051741",
                      "Eagle County": "USW00023063", "Fruita": "USC00053146",
                      "Glenwood Springs": "USC00053359", "Grand Junction": "USC00053489",
                      "Grand Lake": "USC00053500", "Green Mt Dam": "USC00053592",
                      "Kremmling": "USC00054664", "Meredith": "USC00055507",
                      "Rifle": "USC00057031", "Yampa": "USC00059265"}


# filepaths
processedDir = os.path.dirname(os.path.dirname(__file__)) + "/processed/{}".format(dataRepo)
plotsDir = os.path.dirname(os.path.dirname(__file__)) + "/plots"


# plot the raw monthly data
def PlotRawMonthly():
    # for each weather variable...
    for weatherVar in ["precip", "temp"]:
        # are we plotting precip or temp
        if weatherVar == "precip":
            ylabel = "Total Monthly Precipitation [m]"
            dataVar = "PRCP"
        else:
            ylabel = "Average Monthly Temperature [$^\circ$C]"
            dataVar = "TAVG"
        # make the plot
        monthlyComparisonPlot, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all", sharey="all")
        monthlyComparisonPlot.supxlabel("Date"), monthlyComparisonPlot.supylabel(ylabel)
        for i, axis in enumerate(axes.flat):
            # pre-formatting
            stationIndex = rawMonthlyDF["NAME"] == stations[i]
            axis.grid()
            axis.set(title=stations[i])
            if weatherVar == "precip":
                axis.set_ylim(bottom=0, top=0.25)
            else:
                axis.set_ylim(bottom=-20, top=30)
            # plot the data from each ID associated with each station
            for ID in set(rawMonthlyDF.loc[rawMonthlyDF["NAME"] == stations[i], "ID"]):
                idIndex = rawMonthlyDF["ID"] == ID
                # format the date
                dates = rawMonthlyDF.loc[stationIndex & idIndex, ["YEAR", "MONTH"]].astype(str).apply("-".join, axis=1).values
                dates = [dt.datetime.strptime(d, "%Y-%b").date() for d in dates]
                # grab the data
                data2plot = rawMonthlyDF.loc[stationIndex & idIndex, dataVar].values
                # plot
                axis.plot(dates, data2plot, marker=".", label=ID)
            axis.legend()

            # post formatting, show, save
            plt.tight_layout()
            monthlyComparisonPlot.savefig(plotsDir + r"/obs/{}_RawMonthly{}_byStationAndArea.svg".format(repoName, weatherVar.capitalize()))
            plt.close()



# plot to show how well stations have been unbiased
def PlotBiases():
    months = sorted(set(rawMonthlyDF["MONTH"]), key=lambda x: dt.datetime.strptime(x, "%b"))
    for station in stations:
        stationIndex = rawMonthlyDF["NAME"] == station
        sIDs = list(set(biasDF.loc[biasDF["Primary"] == primaryStationDict[station], "Secondary"].values))
        for month in months:
            monthIndex = rawMonthlyDF["MONTH"] == month
            biasPlots, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 9))
            biasPlots.suptitle("{} {}".format(station, month))
            for i, axis in enumerate(axes.flat):
                if i < 2:
                    dataVar = "PRCP"
                    scaleVar = "PRCP Scaling"
                    poffsetVar = "PRCP Primary Offset"
                    soffsetVar = "PRCP Secondary Offset"
                    yLabel = "Total Monthly Precipitation [m]"
                else:
                    dataVar = "TAVG"
                    scaleVar = "TAVG Scaling"
                    poffsetVar = "TAVG Primary Offset"
                    soffsetVar = "TAVG Secondary Offset"
                    yLabel = "Average Monthly Temperature [$^\circ$C]"
                axis.grid()
                # bias correction
                if i % 2 == 0:
                    axis.set(title="Bias Correction")
                    axis.set(xlabel="Year", ylabel=yLabel)
                    # plot the primary, secondary data with the bias correction
                    axis.plot(rawMonthlyDF.loc[monthIndex & stationIndex & (rawMonthlyDF["ID"] == primaryStationDict[station]), "YEAR"].values,
                              rawMonthlyDF.loc[monthIndex & stationIndex & (rawMonthlyDF["ID"] == primaryStationDict[station]), dataVar].values, marker=".", label=primaryStationDict[station])
                    for otherID in sIDs:
                        secondaryIndex = rawMonthlyDF["ID"] == otherID
                        scale = biasDF.loc[(biasDF["Month"] == month) & (biasDF["Secondary"] == otherID), scaleVar].values[0]
                        pOffset = biasDF.loc[(biasDF["Month"] == month) & (biasDF["Secondary"] == otherID), poffsetVar].values[0]
                        sOffset = biasDF.loc[(biasDF["Month"] == month) & (biasDF["Secondary"] == otherID), soffsetVar].values[0]
                        biasCorrectedData = scale * (rawMonthlyDF.loc[monthIndex & stationIndex & secondaryIndex, dataVar].values - sOffset) + pOffset
                        axis.plot(rawMonthlyDF.loc[monthIndex & stationIndex & secondaryIndex, "YEAR"].values, biasCorrectedData, marker=".", label=otherID + " w/o bias")
                    axis.legend()

                # Q-Q plot of unbiased secondary stations against primary along data normal dist
                else:
                    axis.set(title="Q-Q Plot of Secondaries, Bias-Corrected to Primary".format())
                    axis.set(xlabel="Theoretical Q", ylabel="Sample Q")
                    primaryData = rawMonthlyDF.loc[monthIndex & stationIndex & (rawMonthlyDF["ID"] == primaryStationDict[station]), dataVar].values
                    # plot the Q-Q plot
                    for j, otherID in enumerate(sIDs):
                        secondaryIndex = rawMonthlyDF["ID"] == otherID
                        scale = biasDF.loc[(biasDF["Month"] == month) & (biasDF["Secondary"] == otherID), scaleVar].values[0]
                        pOffset = biasDF.loc[(biasDF["Month"] == month) & (biasDF["Secondary"] == otherID), poffsetVar].values[0]
                        sOffset = biasDF.loc[(biasDF["Month"] == month) & (biasDF["Secondary"] == otherID), soffsetVar].values[0]
                        biasCorrectedData = scale * (rawMonthlyDF.loc[monthIndex & stationIndex & secondaryIndex, dataVar].values - sOffset) + pOffset
                        if all(np.isnan(biasCorrectedData)):
                            continue
                        pp = sm.ProbPlot(data=biasCorrectedData,  dist=stats.norm, a=1/2, loc=pd.Series(primaryData, dtype=float).mean(),
                                         scale=pd.Series(primaryData, dtype=float).std())
                        pp.qqplot(ax=axis, markerfacecolor="C{}".format(j+1), markeredgecolor="C{}".format(j+1))
                        sm.qqline(ax=axis, line="45", fmt="k-")

            # show, save, close
            plt.tight_layout()
            biasPlots.savefig(plotsDir + r"/obs/{}_BiasCorrection_{}{}.svg".format(repoName, station.replace(" ", ""), month))
            plt.close()



# comparison plot for the combined dataset vs. the yearly data with bias correction
def PlotCombinedVSYearly():
    # get the stations
    months = sorted(set(monthlyDF["MONTH"]), key=lambda x: dt.datetime.strptime(x, "%b"))

    # for each weather variable...
    for weatherVar in ["precip", "temp"]:
        # are we plotting precip or temp
        if weatherVar == "precip":
            ylabel = "Total Yearly Precipitation [m]"
            dataVar = "PRCP"
            scaleVar = "PRCP Scaling"
            poffsetVar = "PRCP Primary Offset"
            soffsetVar = "PRCP Secondary Offset"
        else:
            ylabel = "Average Yearly Temperature [$^\circ$C]"
            dataVar = "TAVG"
            scaleVar = "TAVG Scaling"
            poffsetVar = "TAVG Primary Offset"
            soffsetVar = "TAVG Secondary Offset"

        # make the plot
        combinedYearlyComparisonPlot, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all", sharey="all")
        combinedYearlyComparisonPlot.supxlabel("Date"), combinedYearlyComparisonPlot.supylabel(ylabel)
        for i, axis in enumerate(axes.flat):
            # pre-formatting
            comboStationIndex = monthlyDF["NAME"] == stations[i]
            monthlyStationIndex = rawMonthlyDF["NAME"] == stations[i]
            axis.grid()
            axis.set(title=stations[i])
            if weatherVar == "precip":
                axis.set_ylim(bottom=0, top=1)
            else:
                axis.set_ylim(bottom=-10, top=20)
            # plot the data from each ID associated with each station
            for ID in set(rawMonthlyDF.loc[rawMonthlyDF["NAME"] == stations[i], "ID"]):
                idIndex = rawMonthlyDF["ID"] == ID
                dates = list(set(rawMonthlyDF.loc[monthlyStationIndex & idIndex, "YEAR"].values))
                if ID in set(biasDF["Primary"].values):
                    if weatherVar == "precip":
                        yearlyData2plot = [np.nansum(rawMonthlyDF.loc[monthlyStationIndex & idIndex & (rawMonthlyDF["YEAR"] == year), dataVar].values) for year in dates]
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            yearlyData2plot = [np.nanmean(rawMonthlyDF.loc[monthlyStationIndex & idIndex & (rawMonthlyDF["YEAR"] == year), dataVar].values) for year in dates]
                    axis.plot(dates, yearlyData2plot, marker=".", label=ID)
                else:
                    monthlyData2plot = np.full(shape=(len(dates), len(months)), fill_value=np.NaN)
                    for m, month in enumerate(months):
                        scale = biasDF.loc[(biasDF["Secondary"] == ID) & (biasDF["Month"] == month), scaleVar].values[0]
                        primaryOffset = biasDF.loc[(biasDF["Secondary"] == ID) & (biasDF["Month"] == month), poffsetVar].values[0]
                        secondaryOffset = biasDF.loc[(biasDF["Secondary"] == ID) & (biasDF["Month"] == month), soffsetVar].values[0]
                        monthlyData2plot[:, m] = scale * (rawMonthlyDF.loc[monthlyStationIndex & idIndex & (rawMonthlyDF["MONTH"] == month), dataVar].values - secondaryOffset) + primaryOffset
                    if weatherVar == "precip":
                        axis.plot(dates, np.nansum(monthlyData2plot, axis=1).reshape(-1, 1), marker=".", label=ID + " + bias")
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            axis.plot(dates, np.nanmean(monthlyData2plot, axis=1).reshape(-1, 1), marker=".", label=ID + " + bias")
            # plot the data from the combined NOAA data
            comboYears = sorted(set(monthlyDF.loc[comboStationIndex, "YEAR"].values))
            comboData2Plot = [np.NaN] * len(comboYears)
            for j, year in enumerate(comboYears):
                if weatherVar == "precip":
                    comboData2Plot[j] = np.nansum(monthlyDF.loc[comboStationIndex & (monthlyDF["YEAR"] == year), "PRCP"].values)
                else:
                    comboData2Plot[j] = np.nanmean(monthlyDF.loc[comboStationIndex & (monthlyDF["YEAR"] == year), "TAVG"].values)
            axis.plot(comboYears, comboData2Plot, color="black", marker=".", label="Combined")
            axis.legend()

        # post formatting, show, save
        plt.tight_layout()
        combinedYearlyComparisonPlot.savefig(plotsDir + r"/obs/{}_CompareCombinedVSYearly_{}.svg".format(repoName, weatherVar.capitalize()))
        plt.close()


# plot seasonal (monthly) spatial correlation matrices for precipitation and temperature
def PlotMonthlySpatialCorrelations():
    #cmapCorr = plt.get_cmap("gnuplot", 11)
    cmapCorr = plt.get_cmap("gnuplot")
    # set up the valid years to look at
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    gmmhmmYears = GMMHMMDict["precipDF"].index.values
    validYears = list(range(min(gmmhmmYears), max(gmmhmmYears)+1))
    for month in months:
        monthlyMonthIdx = monthlyDF["MONTH"] == month 
        stationsPrecipDF, stationsTempDF = pd.DataFrame(columns=stations), pd.DataFrame(columns=stations)
        for station in stations:
            monthlyStationIdx = monthlyDF["NAME"] == station
            precips, temps = [], []
            # look at the years we start having some data for 
            for year in validYears:
                monthlyYearIdx = monthlyDF["YEAR"] == year
                precipEntry = monthlyDF.loc[monthlyStationIdx & monthlyMonthIdx & monthlyYearIdx, "PRCP"]
                tempEntry = monthlyDF.loc[monthlyStationIdx & monthlyMonthIdx & monthlyYearIdx, "TAVG"]
                precips.append(np.NaN if precipEntry.empty else precipEntry.values[0])
                temps.append(np.NaN if tempEntry.empty else tempEntry.values[0])
            stationsPrecipDF[station], stationsTempDF[station] = precips, temps
        stationsPrecipDF.index, stationsTempDF.index = validYears, validYears
        
        # plot correlation matrices for the historic monthly data, individually
        for wvar in ["precip", "tavg"]:
            monthlyHistoricCorrelation = stationsPrecipDF.corr(method="pearson") if wvar == "precip" else stationsTempDF.corr(method="pearson")
            HistoricCorrPlot, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), sharex="all", sharey="all")
            histCorrColors = axes.matshow(monthlyHistoricCorrelation.values, vmin=0, vmax=1, cmap=cmapCorr)
            # for i, station1 in enumerate(stations):
            #     for j, station2 in enumerate(stations):
            #         plt.text(j, i, '{:.2f}'.format(monthlyHistoricCorrelation.at[station1, station2]), ha='center', va='center', backgroundcolor=[0, 0, 0, 0.25], color="white")
            HistoricCorrPlot.colorbar(histCorrColors, label="{} {} Spatial Correlation [Pearson, -]".format(repoName, month))
            plt.xticks(range(len(stations)), stations, rotation=75)
            plt.yticks(range(len(stations)), stations, rotation=0)
            HistoricCorrPlot.savefig(plotsDir + r"/obs/{}_Monthly{}SpatialCorrelation_{}.svg".format(repoName, wvar.capitalize(), month))
            plt.close()
    
    # plot the aggregated monthly temperature correlation matrix
    aggTempDF = pd.DataFrame(columns=stations)
    for station in stations:
        aggStationIdx = monthlyDF["NAME"] == station
        aggData = []
        for year in validYears:
            aggYearIdx = monthlyDF["YEAR"] == year
            for month in months:
                aggMonthIdx = monthlyDF["MONTH"] == month
                aggTempEntry = monthlyDF.loc[aggStationIdx & aggYearIdx & aggMonthIdx, "TAVG"]
                aggData.append(np.NaN if aggTempEntry.empty else aggTempEntry.values[0])
        aggTempDF[station] = aggData
    aggHistoricCorrelation = aggTempDF.corr(method="pearson")
    AggHistoricCorrPlot, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), sharex="all", sharey="all")
    aggHistCorrColors = axes.matshow(aggHistoricCorrelation.values, vmin=0, vmax=1, cmap=cmapCorr)
    # for i, station1 in enumerate(stations):
    #     for j, station2 in enumerate(stations):
    #         plt.text(j, i, '{:.2f}'.format(aggHistoricCorrelation.at[station1, station2]), ha='center', va='center', backgroundcolor=[0, 0, 0, 0.25], color="white")
    AggHistoricCorrPlot.colorbar(aggHistCorrColors, label="{} Monthly Temp Spatial Correlation [Pearson, -]".format(repoName))
    plt.xticks(range(len(stations)), stations, rotation=75)
    plt.yticks(range(len(stations)), stations, rotation=0)
    AggHistoricCorrPlot.savefig(plotsDir + r"/obs/{}_MonthlyTavgSpatialCorrelation.svg".format(repoName))
    plt.close()


# investigating the autocorrelation and partial autocorrelation to determine Markovian structure of the formatted data
def PlotUCRBMarkovian():    
    seqMonthlyDF, transformedAnnualDF = GMMHMMDict["monthlyDF"], GMMHMMDict["precipDF"]
    for timeScale in ["monthly", "annual"]:
        for corr in ["acf", "pacf"]:
            markovianPlot, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all", sharey="all")
            markovianPlot.suptitle("{} for {} Precip".format(corr.upper(), timeScale.capitalize()))
            markovianPlot.supxlabel("Lag"), markovianPlot.supylabel("{} [-]".format(corr.upper()))
            for i, axis in enumerate(axes.flat):
                # pre-processing
                axis.grid()

                # data for the autoregression
                dataAR = seqMonthlyDF[stations[i]].values if timeScale == "monthly" else transformedAnnualDF[stations[i]].values
                # ACF or PACF
                if corr == "acf":
                    plot_acf(ax=axis, x=dataAR, color="deepskyblue", vlines_kwargs={"color": "deepskyblue"})
                    fitAR1 = AutoReg(dataAR, lags=[1]).fit()
                    plot_acf(ax=axis, x=fitAR1.resid, color="black", vlines_kwargs={"color": "deepskyblue"})
                else:
                    plot_pacf(ax=axis, x=dataAR, method="ywm", color="deepskyblue", vlines_kwargs={"color": "black"})
                axis.set(title=stations[i], xlim=(-0.25, 11.25))
                axis.hlines(0, xmin=0, xmax=11.25, color="red")
            plt.tight_layout()
            markovianPlot.savefig(plotsDir + r"/gmmhmm/{}/{}_MarkovianStructure_{}{}.svg".format(dataRepo, repoName, timeScale.capitalize(), corr.upper()))
            plt.close()


# plot the statistics relavant to the GMMHMM validation 
def PlotGMMHMMStatistics():
    pDict = GMMHMMDict
    # get some basic info about stations, months
    years = pDict["precipDF"].index.values

    # plot the ACF and PACF for the discovered states to check if Markovian
    if len(set(pDict["hiddenStates"])) > 1:
        hiddenStateTemporalityPlot, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 9), sharex="all")
        hiddenStateTemporalityPlot.suptitle("Markovian Structure of Fit HMM Hidden States")
        hiddenStateTemporalityPlot.supxlabel("Lag")
        for i, axis in enumerate(axes.flat):
            # pre-plot stuff
            axis.grid()

            # ACF or PACF
            if i == 0:
                plot_acf(ax=axis, x=pDict["hiddenStates"], vlines_kwargs={"color": "black"})
                axis.set(ylabel="ACF [-]", xlim=(-0.25, 16.25))
                axis.hlines(0, xmin=0, xmax=24, color="red")
            else:
                plot_pacf(ax=axis, x=pDict["hiddenStates"], method="ywm", vlines_kwargs={"color": "black"})
                axis.set(ylabel="PACF [-]", xlim=(-0.25, 16.25))
                axis.hlines(0, xmin=0, xmax=24, color="red")
        plt.tight_layout()
        hiddenStateTemporalityPlot.savefig(plotsDir + r"/gmmhmm/{}/{}_GMMHMMFit_MarkovianStructure.svg".format(dataRepo, repoName))
        plt.close()

    # plot the transition probability matrix
    numStates = pDict["model"].n_components
    TProbPlot, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), sharex="all", sharey="all")
    TProbPlot.suptitle("Transition Probabilities from HMM")
    transProbColors = axes.matshow(pDict["tProbs"], vmin=0, vmax=1, cmap="gnuplot")
    for i in range(pDict["tProbs"].shape[0]):
        for j in range(pDict["tProbs"].shape[1]):
            # text for the values
            axes.text(j, i, '{:.3f}'.format(pDict["tProbs"][i, j]), ha='center', va='center', backgroundcolor=[0, 0, 0, 0.25], color="white")
    # standardize labels, colors
    axes.set(xticks=range(numStates), xticklabels=["To State {}".format(s) for s in range(numStates)],
             yticks=range(numStates), yticklabels=["From State {}".format(s) for s in range(numStates)])
    TProbPlot.colorbar(transProbColors, label="Probability [-]")
    plt.tight_layout()
    TProbPlot.savefig(plotsDir + r"/gmmhmm/{}/{}_GMMHMMTransitionProbabilities.svg".format(dataRepo, repoName))
    plt.close()

    # plot the hidden states as a function of date
    hmmColors = ["black"]
    hiddenStateTimeSeries, axis = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all", sharey="all")
    hiddenStateTimeSeries.suptitle("Hidden States with Annual Totals")
    hiddenStateTimeSeries.supxlabel("Date"), hiddenStateTimeSeries.supylabel("Annual Total Precip [m]")
    for i, axis in enumerate(axis.flat):
        # axis.grid()
        axis.text(0.5, 0.65, stations[i], fontsize="large", transform=axis.transAxes)
        axis.plot(years, 10 ** pDict["precipDF"][stations[i]].values, color="black", marker=".", linestyle="-")
        for s in range(numStates):
            stateIndex = pDict["hiddenStates"] == s
            axis.plot(np.array(years)[stateIndex], 10 ** pDict["precipDF"][stations[i]].values[stateIndex], color=hmmColors[s], marker="o", linestyle="None")
    plt.tight_layout()
    hiddenStateTimeSeries.savefig(plotsDir + r"/gmmhmm/{}/{}_HiddenStateTimeSeries.svg".format(dataRepo, repoName))
    plt.close()

    # Q-Q plots, to check that the points make sense to be distributed as the GMMHMM suggests
    QQPlot, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9))
    QQPlot.suptitle("Q-Q Plot of States' Log-Normal Distributions vs. Annual Data")
    QQPlot.supxlabel("Theoretical Quantiles [-]"), QQPlot.supylabel("Data Quantiles [-]")
    for i, axis in enumerate(axes.flat):
        axis.grid()
        axis.set(title=stations[i])
        mus = [pDict[stations[i]]["means"][s] for s in range(numStates)]
        stds = [pDict[stations[i]]["stds"][s] for s in range(numStates)]
        # get data, make Q-Q
        for s in range(numStates):
            stateIndexing = pDict["hiddenStates"] == s
            data = pDict["precipDF"][stations[i]].values[stateIndexing]
            PP = sm.ProbPlot(data=data, dist=stats.norm, a=1/2, loc=mus[s], scale=stds[s])
            PP.qqplot(ax=axis, markerfacecolor=hmmColors[s], markeredgecolor=hmmColors[s], alpha=0.5, xlabel=None, ylabel=None)
        sm.qqline(ax=axis, line="45", fmt="k-")
    plt.tight_layout()
    QQPlot.savefig(plotsDir + r"/gmmhmm/{}/{}_QQ_GMMHMMFit.svg".format(dataRepo, repoName))
    plt.close()

    # plot correlation matrices for the historic data
    cmapCorr = plt.get_cmap("gnuplot", 11)
    actualHistoricCorrelation = pDict["monthlyDF"].corr(method="pearson")
    HistoricCorrPlot, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), sharex="all", sharey="all")
    histCorrColors = axes.matshow(actualHistoricCorrelation.values, vmin=0, vmax=1, cmap=cmapCorr)
    # for i, station1 in enumerate(stations):
    #     for j, station2 in enumerate(stations):
    #         plt.text(j, i, '{:.2f}'.format(actualHistoricCorrelation.at[station1, station2]), ha='center', va='center', backgroundcolor=[0, 0, 0, 0.25], color="white")
    HistoricCorrPlot.colorbar(histCorrColors, label="{} Monthly Spatial Correlation [Pearson, -]".format(repoName))
    plt.xticks(range(len(stations)), stations, rotation=75)
    plt.yticks(range(len(stations)), stations, rotation=0)
    HistoricCorrPlot.savefig(plotsDir + r"/gmmhmm/{}/{}_SpatialCorrelation.svg".format(dataRepo, repoName))
    plt.close()

    if ("NASA" in dataRepo):
        # plot a covariance/correlation vs. station distance curve
        CorrVsDistPlot, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 9), sharex="all")
        CorrVsDistPlot.suptitle("Multivariate Normal Covariance-->Correlation Values vs. Station Distance")
        CorrVsDistPlot.supxlabel("Station Distance [km]")
        ds, corrs = [], []
        for i, axis in enumerate(axes.flat):
            axis.grid()
            if i == 0:
                for s1, station1 in enumerate(stations):
                    station1Idx = monthlyDF["NAME"] == station1
                    for s2, station2 in enumerate(stations):
                        if s2 >= s1: continue
                        station2Idx = monthlyDF["NAME"] == station2
                        lat1, lon1 = list(set(monthlyDF.loc[station1Idx, "LAT"].values))[0], list(set(monthlyDF.loc[station1Idx, "LON"].values))[0]
                        lat2, lon2 = list(set(monthlyDF.loc[station2Idx, "LAT"].values))[0], list(set(monthlyDF.loc[station2Idx, "LON"].values))[0]
                        elev1, elev2 = list(set(monthlyDF.loc[station1Idx, "ELEV"].values))[0], list(set(monthlyDF.loc[station2Idx, "ELEV"].values))[0]
                        d = CalculateGeographicDistance([lat1, lon1, elev1], [lat2, lon2, elev2])
                        ds.append(d), corrs.append(pDict["model"].covars_[0][0][s1][s2] / (pDict[station1]["stds"][0] * pDict[station2]["stds"][0]))
                axis.scatter(ds, corrs, color="black", label="Data")
                axis.scatter(0., 1., color="red", label="Self-Correlation (Diagonals)")
                dspace = np.linspace(min(ds), max(ds), 100)
                expDecayParams = [*pDict["corrdist"]["expDecayParams"]]
                axis.plot(dspace, expDecayParams[0]*np.exp(-expDecayParams[1] * dspace) + expDecayParams[2], 
                          color="black", linestyle="dashed", label="fit: {:.3f}*exp(-{:.3f}d) + {:.3f}".format(expDecayParams[0], expDecayParams[1], expDecayParams[2]))
                axis.set_ylim(0, 1.05)
                axis.set_ylabel("Correlation Value [Pearson, -]")
                axis.legend()
            if i == 1:
                axis.scatter(ds, pDict["corrdist"]["resids"], color="black", 
                             label="Data | mean: {:.3f} | std: {:.3f}".format(np.nanmean(pDict["corrdist"]["resids"]), np.nanstd(pDict["corrdist"]["resids"])))
                axis.scatter(0., 0., color="red", label="Self-Correlation (Diagonals)")
                axis.set_ylabel("Correlation Residuals [-]")
                axis.legend()
        plt.tight_layout()
        CorrVsDistPlot.savefig(plotsDir + r"/gmmhmm/{}/{}_PrecipCorrVsDist.svg".format(dataRepo, repoName))
        plt.close()


# plot the radar AIC plots
def PlotCopulaAICs():
    ptDict = CopulaDict
    AICPlots = plt.figure(figsize=(14, 9))
    AICPlots.suptitle('{} Monthly AIC by Copula'.format(repoName))
    months = list(ptDict.keys())
    indAIC, frkAIC, gauAIC = [np.NaN] * len(months), [np.NaN] * len(months), [np.NaN] * len(months)
    # create the data
    for j, month in enumerate(months):
        fitInfo = ptDict[month]["CopulaDF"]
        indAIC[j] = fitInfo.at["Independence", "AIC"]
        frkAIC[j] = fitInfo.at["Frank", "AIC"]
        gauAIC[j] = fitInfo.at["Gaussian", "AIC"]
    # make sure they wrap
    indAIC = [*indAIC, indAIC[0]]
    frkAIC = [*frkAIC, frkAIC[0]]
    gauAIC = [*gauAIC, gauAIC[0]]
    months = [*months, months[0]]
    monthLabelLoc = np.linspace(0, 2 * np.pi, num=len(months))

    # setting up the plot, title
    axis = AICPlots.add_subplot(1, 1, 1, projection="polar")
    axis.set_ylim([-40., 5.])
    axis.spines["polar"].set_visible(False)

    # actually plotting things
    axis.plot(monthLabelLoc, indAIC, label="Ind.", c="grey")
    axis.plot(monthLabelLoc, frkAIC, label="Frk.", c="blue")
    axis.plot(monthLabelLoc, gauAIC, label="Gau.", c="red")
    axis.set_thetagrids(np.degrees(monthLabelLoc), labels=months)
    axis.legend(loc="upper right")
    plt.tight_layout()
    AICPlots.savefig(plotsDir + r"/copulae/{}/{}_CopulaAICPlots.svg".format(dataRepo, repoName))
    plt.close()
 

# plot the empirical copula with the data's pseudo-observations, and the isolines for each theoretical copula
def PlotCopulae():
    # calculate empirical copula
    def CalculateEmpiricalCopula2D(pseudoObservations, resolution=100):
        # establish the resolution of the empirical copula
        nData, nDim = pseudoObservations.shape[0], pseudoObservations.shape[1]
        u0 = np.linspace(0, 1, resolution)

        # actual calculation
        Cn = np.full(shape=(resolution,)*nDim, fill_value=0.)
        for k in range(nData):
            U1, U2 = pseudoObservations[k, 0], pseudoObservations[k, 1]
            for ii in range(resolution):
                for jj in range(resolution):
                    if U1 <= u0[ii] and U2 <= u0[jj]:
                        Cn[ii, jj] += 1.
        Cn /= nData

        # return the empirical copula (and marginal resolution)
        return Cn, u0    
   

    # approximate the theoretical copula, mainly for plotting
    def ApproximateTheoreticalCopula2D(copula, marginals, useDF=False):
        # empty matrix for the cdf values
        Ctheory = np.full(shape=(len(marginals), len(marginals)), fill_value=np.NaN)

        # fill the theoretical 2D CDF
        for ii, u1 in enumerate(marginals):
            for jj, u2 in enumerate(marginals):
                dataDF = pd.DataFrame(data={"uP": [u1], "uT": [u2]}, dtype=float)
                Cval = copula.cdf(dataDF.values) if useDF else copula.cdf(dataDF.values)
                Ctheory[ii, jj] = Cval[0] if type(Cval) in [list, np.ndarray] else Cval

        # return the theoretical CDF
        return Ctheory

 
    ptDict = CopulaDict
    months = list(ptDict.keys())
    CopPlots, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all", sharey="all")
    CopPlots.suptitle("{} Copulas".format(repoName))
    CopPlots.supxlabel("$U_P$"), CopPlots.supylabel("$U_T$")
    # for each month...
    for i, axis in enumerate(axes.flat):
        month = months[i]
        pObs = np.array([ptDict[month]["PRCP pObs"], ptDict[month]["TAVG pObs"]]).T
        empC, u = CalculateEmpiricalCopula2D(pObs)

        # scatterplot of the residual pseudo-observations
        axis.grid()
        axis.set_title(month)
        axis.scatter(pObs[:, 0], pObs[:, 1], c="black", s=1)
        contourLevels = [(i + 1) / 10 for i in range(10)]
        # empirical copula
        eContour = axis.contour(u, u, empC, levels=contourLevels, colors="black", vmin=0, vmax=1, linewidths=0.75)
        axis.clabel(eContour, inline=True, fontsize=10)
        # the copulas to fit
        copulaFitDF = ptDict[month]["CopulaDF"]
        for family in copulaFitDF.index.values:
            if family == "Independence":
                axis.contour(u, u, ApproximateTheoreticalCopula2D(copulaFitDF.at[family, "Copula"], u),
                             levels=contourLevels, colors="grey", vmin=0, vmax=1, linewidths=0.75, linestyles="dashed")
            elif family == "Frank":
                axis.contour(u, u, ApproximateTheoreticalCopula2D(copulaFitDF.at[family, "Copula"], u),
                             levels=contourLevels, colors="blue", vmin=0, vmax=1, linewidths=0.75, linestyles="dashed")
            elif family == "Gaussian":
                axis.contour(u, u, ApproximateTheoreticalCopula2D(copulaFitDF.at[family, "Copula"], u, useDF=True),
                             levels=contourLevels, colors="red", vmin=0, vmax=1, linewidths=0.75, linestyles="dashed")
    plt.tight_layout()
    CopPlots.savefig(plotsDir + r"/copulae/{}/{}_PseudoObs_and_Copulae.svg".format(dataRepo, repoName))
    plt.close()


# plot CMIP6 bias corrections
def PlotCMIP6BiasCorrections():
    viridis = colormaps["viridis"]
    newcolors = viridis(np.linspace(0, 1, 1000))
    newcolors[:50, :] = np.array([0., 0., 0., 1.])
    newcmp = ListedColormap(newcolors)
    statMetrics = ["t-Test", "MWU", "Levene", "KS"]
    for wvar in ["PRCP", "TAVG"]:
        statsDict = {} 
        # boxplots
        biasCorrDistPlot, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9), sharex="all", sharey="all")
        biasCorrDistPlot.suptitle("CMIP6 {} Bias-Corrected Distributions for: {}".format(wvar, repoName))
        biasCorrDistPlot.supxlabel("Months")
        biasCorrDistPlot.supylabel("PRCP [m]") if wvar == "PRCP" else biasCorrDistPlot.supylabel("TAVG [" + chr(176) + "C]")
        boxWidth = 0.1
        wvarColor = "royalblue" if wvar == "PRCP" else "firebrick"
        for i, axis in enumerate(axes.flat):
            station = stations[i]
            stationStatsDict = {}
            obsStationIdx = obsMonthlyDF["NAME"] == station
            rawStationIdx = rawMonthlyDF["STATION"] == station
            bcStationIdx = monthlyDF["NAME"] == station
            axis.set_title(station)
            axis.grid()
            for m, month in enumerate(months):
                obsMonthIdx = obsMonthlyDF["MONTH"] == month
                rawMonthIdx = rawMonthlyDF["MONTH"] == month
                bcMonthIdx = monthlyDF["MONTH"] == month
                obsVals = obsMonthlyDF.loc[obsStationIdx & obsMonthIdx & obsYearIdx, wvar].values
                rawVals = rawMonthlyDF.loc[rawStationIdx & rawMonthIdx, wvar].values
                bcVals = monthlyDF.loc[bcStationIdx & bcMonthIdx, wvar].values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ttestPValue = stats.ttest_ind(obsVals, bcVals, equal_var=False).pvalue
                    mwuPValue = stats.mannwhitneyu(obsVals, bcVals).pvalue
                    levenePValue = stats.levene(obsVals, bcVals).pvalue
                    ksPValue = stats.ks_2samp(obsVals, bcVals).pvalue
                stationStatsDict[month] = [ttestPValue, mwuPValue, levenePValue, ksPValue]
                axis.boxplot(obsVals, sym=None, positions=[m+1-2*boxWidth], widths=boxWidth, patch_artist=True, 
                             boxprops={"facecolor": "black"}, medianprops={"color": wvarColor}, flierprops={"marker": ".", "markeredgecolor": "black"})
                axis.boxplot(rawVals, sym=None, positions=[m+1], widths=boxWidth, patch_artist=True, 
                             boxprops={"facecolor": "grey"}, medianprops={"color": "black"}, flierprops={"marker": ".", "markeredgecolor": "grey"})
                axis.boxplot(bcVals, sym=None, positions=[m+1+2*boxWidth], widths=boxWidth, patch_artist=True, 
                             boxprops={"facecolor": wvarColor}, medianprops={"color": "black"}, flierprops={"marker": ".", "markeredgecolor": wvarColor})
            statsDict[station] = pd.DataFrame().from_dict(stationStatsDict, orient="index", columns=statMetrics)
            axis.set_xticks(np.arange(1, len(months)+1))
            axis.set_xticklabels(months, rotation=45, ha="right")
            axis.tick_params(axis="x", labelsize=len(months))
        plt.tight_layout()
        if repoPieces[2] == "historical":
            biasCorrDistPlot.savefig(plotsDir + r"/cmip6/historical/{}_{}BiasCorrDistribution.svg".format(repoName, wvar))
        else:
            biasCorrDistPlot.savefig(plotsDir + r"/cmip6/ssp/{}_{}BiasCorrDistribution.svg".format(repoName, wvar))
        plt.close() 

        # stats plots
        if repoPieces[2] == "historical":
            biasCorrStatsPlot, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9), sharex="all", sharey="all")
            biasCorrStatsPlot.suptitle("CMIP6 {} Bias Correction Stats for: {}".format(wvar, repoName))
            biasCorrStatsPlot.supxlabel("Months"), biasCorrStatsPlot.supylabel("Metrics")
            for i, axis in enumerate(axes.flat):
                station = stations[i]
                obsStationIdx = obsMonthlyDF["NAME"] == station
                rawStationIdx = rawMonthlyDF["STATION"] == station
                bcStationIdx = monthlyDF["NAME"] == station
                axis.set_title(station)
                metrics = axis.imshow(statsDict[station].values.transpose(), cmap=newcmp, interpolation="nearest", aspect="auto", vmin=0., vmax=1., origin="lower")
                axis.set_xticks(range(len(months))), axis.set_xticklabels(months, rotation=45, ha="right")
                axis.set_yticks(range(len(statMetrics))), axis.set_yticklabels(statMetrics)
            biasCorrStatsPlot.subplots_adjust(right=0.933)
            cbar_ax = biasCorrStatsPlot.add_axes([0.95, 0.1, 0.0167, 0.8])
            biasCorrStatsPlot.colorbar(metrics, cax=cbar_ax)
            biasCorrStatsPlot.savefig(plotsDir + r"/cmip6/historical/{}_{}BiasCorrStats.svg".format(repoName, wvar))
            plt.close()


# run the program
if __name__ == "__main__":
    # helper function for parallelizing the plotting
    def MultiprocessPlotting(fns, fninputs):
        pross = []
        for i, fn in enumerate(fns):
            p = Process(target=fn, args=fninputs[i])
            p.start()
            pross.append(p)
        for p in pross:
            p.join()
    
    # load in the datasets we'll need
    monthlyDF = pd.read_csv(processedDir + r"/{}_UCRBMonthly.csv".format(repoName)) 
    GMMHMMDict = np.load(processedDir + r"/{}_MultisiteGMMHMMFit.npy".format(repoName), allow_pickle=True).item()
    CopulaDict = np.load(processedDir + r"/{}_CopulaFits.npy".format(repoName), allow_pickle=True).item()
    stations = sorted(set(monthlyDF["NAME"].values))
   
    # plot everything!
    plotFunctionList = [PlotUCRBMarkovian, PlotGMMHMMStatistics, PlotCopulaAICs, PlotCopulae]
    if "NOAA" in dataRepo:
        plotFunctionList.extend([PlotRawMonthly, PlotBiases, PlotCombinedVSYearly, PlotMonthlySpatialCorrelations])
        rawMonthlyDF = pd.read_csv(processedDir + r"/Raw{}_Monthly.csv".format(repoName))
        biasDF = pd.read_csv(processedDir + r"/Raw{}_Biases.csv".format(repoName))
    if "CMIP6" in dataRepo:
        repoPieces = repoName.split("_")
        cmip6Source, cmip6Pathway, cmip6Model = repoPieces[1], repoPieces[2], repoPieces[3]
        plotFunctionList.extend([PlotCMIP6BiasCorrections])
        obsMonthlyDF = pd.read_csv(os.path.dirname(os.path.dirname(__file__)) + r"/processed/NOAA/NOAA_UCRBMonthly.csv")
        cmip6Dir = os.path.dirname(os.path.dirname(__file__)) + r"/cmip6"
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if cmip6Source == "nasa":
            rawDailyDF = pd.read_csv(cmip6Dir + r"/{}/{}/{}/Raw_NASACMIP6_{}_{}_Daily.csv".format(cmip6Source, cmip6Pathway, cmip6Model, cmip6Pathway, cmip6Model))
            cmip6Years = sorted(set(rawDailyDF["YEAR"].values))
            obsYearIdx = (obsMonthlyDF["YEAR"] >= min(cmip6Years)) & (obsMonthlyDF["YEAR"] <= max(cmip6Years))
            rawMonthlyDict = {}
            for station in stations:
                dailyStationIdx = rawDailyDF["STATION"] == station
                for year in cmip6Years:
                    dailyYearIdx = rawDailyDF["YEAR"] == year
                    for month in months:
                        dailyMonthIdx = rawDailyDF["MONTH"] == month
                        nasaMonthlyPRCP = np.nansum(rawDailyDF.loc[dailyStationIdx & dailyYearIdx & dailyMonthIdx, "PRCP"].values)
                        nasaMonthlyTAVG = np.nanmean(rawDailyDF.loc[dailyStationIdx & dailyYearIdx & dailyMonthIdx, "TAVG"].values)
                        rawMonthlyDict[(station, year, month)] = [station, year, month, nasaMonthlyPRCP, nasaMonthlyTAVG]
            rawMonthlyDF = pd.DataFrame().from_dict(rawMonthlyDict, orient="index", columns=["STATION", "YEAR", "MONTH", "PRCP", "TAVG"])
        else:
            cmip6Forcing, cmip6Downscale = repoPieces[4], repoPieces[5]
            cmip6Years = [1980, 2019] if repoPieces[2] == "historical" else [2020, 2059]
            obsYearIdx = (obsMonthlyDF["YEAR"] >= cmip6Years[0]) & (obsMonthlyDF["YEAR"] <= cmip6Years[1])
            rawMonthlyPRCP = pd.read_csv(cmip6Dir + r"/{}/{}/{}/MonthlyPRCP_{}{}_{}_{}.csv".format(cmip6Source, cmip6Pathway, cmip6Model, cmip6Forcing, cmip6Downscale, cmip6Years[0], cmip6Years[1]))
            rawMonthlyTMIN = pd.read_csv(cmip6Dir + r"/{}/{}/{}/MonthlyTMIN_{}{}_{}_{}.csv".format(cmip6Source, cmip6Pathway, cmip6Model, cmip6Forcing, cmip6Downscale, cmip6Years[0], cmip6Years[1]))
            rawMonthlyTMAX = pd.read_csv(cmip6Dir + r"/{}/{}/{}/MonthlyTMAX_{}{}_{}_{}.csv".format(cmip6Source, cmip6Pathway, cmip6Model, cmip6Forcing, cmip6Downscale, cmip6Years[0], cmip6Years[1]))
            rawMonthlyDict = {}
            for station in stations:
                prcpStationIdx, tminStationIdx, tmaxStationIdx = rawMonthlyPRCP["STATION"] == station, rawMonthlyTMIN["STATION"] == station, rawMonthlyTMAX["STATION"] == station
                for year in range(cmip6Years[0], cmip6Years[1]+1):
                    prcpYearIdx, tminYearIdx, tmaxYearIdx = rawMonthlyPRCP["YEAR"] == year, rawMonthlyTMIN["YEAR"] == year, rawMonthlyTMAX["YEAR"] == year
                    for month in months:
                        prcpMonthIdx, tminMonthIdx, tmaxMonthIdx = rawMonthlyPRCP["MONTH"] == month, rawMonthlyTMIN["MONTH"] == month, rawMonthlyTMAX["MONTH"] == month
                        ornlMonthlyPRCP = rawMonthlyPRCP.loc[prcpStationIdx & prcpYearIdx & prcpMonthIdx, "PRCP"].astype(float).values[0]
                        ornlMonthlyTAVG = np.nanmean([rawMonthlyTMIN.loc[tminStationIdx & tminYearIdx & tminMonthIdx, "TMIN"].astype(float).values[0], 
                                                      rawMonthlyTMAX.loc[tmaxStationIdx & tmaxYearIdx & tmaxMonthIdx, "TMAX"].astype(float).values[0]])
                        rawMonthlyDict[(station, year, month)] = [station, year, month, ornlMonthlyPRCP, ornlMonthlyTAVG]
            rawMonthlyDF = pd.DataFrame().from_dict(rawMonthlyDict, orient="index", columns=["STATION", "YEAR", "MONTH", "PRCP", "TAVG"])
        obsYearIdx = obsYearIdx if repoPieces[2] == "historical" else ((obsMonthlyDF["YEAR"] >= 1941) & (obsMonthlyDF["YEAR"] <= 2022))
    plotFunctionInputList = [()] * len(plotFunctionList)
    MultiprocessPlotting(plotFunctionList, plotFunctionInputList)

