import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import scipy
import math


def validate_gmmhmm_states(dp: str, min_states: int, max_states: int, lls: list[float], aics: list[float], bics: list[float]) -> None:
    """
    Validation figure for confirming the best-fitting number of states
    for the Gaussian mixture model hidden Markov model that represents
    precipitation

    Parameters
    ----------
    dp: str
        Filepath for saving the validation figure
    min_states: int
        The minimum number of attempted hidden states in fit
    max_states: int
        The maximum number of attempted hidden states in fit 
    lls: list[float]
        Log-likelihood calculation for each fit GMMHMM by number of states
    aics: list[float]
        AIC calculation for each fit GMMHMM by number of states
    bics: list[float]
        BIC calculation for each fit GMMHMM by number of states 
    """
    
    len_states = len(np.arange(min_states, max_states + 1))
    if len_states > len(lls):
        lls.extend([np.nan] * (len_states - len(lls)))
        aics.extend([np.nan] * (len_states - len(aics)))
        bics.extend([np.nan] * (len_states - len(bics)))
    num_states_fig, axis = plt.subplots()
    axis.grid() 
    axis.plot(np.arange(min_states, max_states + 1), aics, color="blue", marker="o", label="AIC")
    axis.plot(np.arange(min_states, max_states + 1), bics, color="green", marker="o", label="BIC")
    axis2 = axis.twinx()
    axis2.plot(np.arange(min_states, max_states + 1), lls, color="orange", marker="o", label="LL")
    axis.legend(handles=axis.lines + axis2.lines)
    axis.set_title("Validation of GMMHMM Best-Fitting Number of States")
    axis.set_xlabel("# States")
    axis.set_ylabel("Criterion Value [-, lower is better]")
    axis2.set_ylabel("Log-Likelihood [-, higher is better]")
    plt.tight_layout()
    num_states_fig.savefig("{}Validate_PrecipGMMHMM_NumStates.svg".format(dp))
    plt.close()


def validate_explore_pt_dependence(dp: str, pt_data: pd.DataFrame) -> None:
    """
    Validation figure for exploring the Kendall and Spearman correlation 
    coefficients between precipitation and temperature. Significant
    positive or negative correlations implies a need for a copula to
    represent the conditional relationship between them.

    Parameters
    ----------
    dp: str
        Filepath for saving the validation figure
    pt_data: pd.DataFrame
        The precipitation and temperature data, as a DataFrame
    """
    
    sites = sorted(set(pt_data["SITE"].values))
    full_years = [y for y in range(np.nanmin(pt_data["YEAR"].values), np.nanmax(pt_data["YEAR"].values)+1)]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # spatially-averaged correlation between precipitation and temperature
    spatial_corr_df = pd.DataFrame({"Kendall": np.NaN, "Spearman": np.NaN}, index=months)
    spatial_data_dict = {month: {"PRECIP": [], "TEMP": []} for month in months}
    for month in months:
        month_index = pt_data["MONTH"] == month
        for year in full_years:
            year_index = pt_data["YEAR"] == year
            ps = pt_data.loc[month_index & year_index, "PRECIP"].values
            p = np.nan if np.all(np.isnan(ps)) else np.nanmean(ps)
            ts = pt_data.loc[month_index & year_index, "TEMP"].values
            t = np.nan if np.all(np.isnan(ts)) else np.nanmean(ts)
            spatial_data_dict[month]["PRECIP"].append(p)
            spatial_data_dict[month]["TEMP"].append(t)
        spatial_corr_df.at[month, "Kendall"] = pd.DataFrame({"PRECIP": spatial_data_dict[month]["PRECIP"], 
                                                             "TEMP": spatial_data_dict[month]["TEMP"]}).corr(method="kendall")["PRECIP"]["TEMP"]
        spatial_corr_df.at[month, "Spearman"] = pd.DataFrame({"PRECIP": spatial_data_dict[month]["PRECIP"], 
                                                              "TEMP": spatial_data_dict[month]["TEMP"]}).corr(method="spearman")["PRECIP"]["TEMP"]
    
    # kendall/spearman correlation metric plot
    sa_corr_fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    sa_corr_fig.suptitle("Correlation of Spatially-Averaged Precip/Temp Data by Month")
    sa_corr_fig.supxlabel("Month"), sa_corr_fig.supylabel("Correlation Coefficient [-]")
    axis.grid()
    axis.set_ylim(-1, 1)
    axis.set_xticks(range(len(months)))
    axis.set_xticklabels(months, rotation=45)
    axis.hlines(0, xmin=0, xmax=11, colors="black", linestyles="dashed")
    axis.plot(range(len(months)), spatial_corr_df["Kendall"], marker="o", label=r"Kendall $\tau$")
    axis.plot(range(len(months)), spatial_corr_df["Spearman"], marker="o", label=r"Spearman $\rho$")
    axis.legend()
    plt.tight_layout()
    sa_corr_fig.savefig("{}Validate_ExploreCorrelation_PT_MonthlySpatialAverage.svg".format(dp))
    plt.close()
     
    # plot scatterplot of spatially averaged precipitation and temperature
    pt_dist_fig = plt.figure(figsize=(14, 9))
    pt_dist_fig.supxlabel("Precipitation"), sa_corr_fig.supylabel("Temperature")
    sub_figs = pt_dist_fig.subfigures(3, 4)
    for i, sub_fig in enumerate(sub_figs.flat):
        axes = sub_fig.subplots(2, 2, gridspec_kw={"width_ratios": [4, 1], "height_ratios": [1, 3]})
        sub_fig.subplots_adjust(wspace=0, hspace=0)
        month = months[i]
        for j, axis in enumerate(axes.flat):
            if j == 0:
                axis.hist(spatial_data_dict[month]["PRECIP"], density=True, color="black")
                axis.set(xticks=[], yticks=[])
            if j == 1:
                axis.axis("off")
                axis.text(0.5, 0.5, month, transform=axis.transAxes, va="center", ha="center")
            if j == 2:
                axis.scatter(spatial_data_dict[month]["PRECIP"], spatial_data_dict[month]["TEMP"], marker="o", facecolors="none", edgecolors="black")
            if j == 3:
                axis.hist(spatial_data_dict[month]["TEMP"], density=True, color="black", orientation="horizontal")
                axis.set(xticks=[], yticks=[])
    pt_dist_fig.savefig("{}Validate_Distribution_PT_MonthlySpatialAverage.svg".format(dp))
    plt.close()


def validate_pt_acf(dp: str, pt_dict: dict, lag: int) -> None:
    """
    Validation figure for ACF fits on precipitation and temperature,
    as small multiples by month

    Parameters
    ----------
    dp: str
        Filepath for saving the validation figure
    pt_dict: pd.DataFrame
        The precipitation and temperature data organized by month, as a dict
    lag: int
        The lag considered in the autocorrelation function
    """
    
    for weather_var in ["PRECIP", "TEMP"]:
        weather_color = "royalblue" if weather_var == "PRECIP" else "firebrick"
        acf_fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), sharex="all", sharey="all")
        acf_fig.suptitle("{} ACF from AR({}) | Color=ACF from Raw Data, Black=ACF from Residuals".format(weather_var.capitalize(), lag))
        acf_fig.supxlabel("Lag [-]"), acf_fig.supylabel("ACF [-]")
        months = list(pt_dict.keys())
        for i, axis in enumerate(axes.flat):
            axis.grid()
            plot_acf(ax=axis, x=pt_dict[months[i]][weather_var], color=weather_color, vlines_kwargs={"color": weather_color, "label": None})
            plot_acf(ax=axis, x=pt_dict[months[i]][weather_var + " ARFit"].resid, color="black", vlines_kwargs={"color": "grey", "label": None})
            axis.set(title=months[i])
        plt.tight_layout()
        acf_fig.savefig("{}Validate_{}_ACF.svg".format(dp, weather_var.capitalize()))
        plt.close()


def validate_pt_stationarity(dp: str, pt_dict: dict, groups: int) -> None:
    """
    Validation figure for checking the stationarity of the precipitation
    and temperature residuals through the Mann-Whitney U test

    Parameters
    ----------
    dp: str
        Filepath for saving the validation figure
    pt_dict: pd.DataFrame
        The precipitation and temperature data organized by month, as a dict
    groups: int
        Number of groups to consider for the stationarity test
    """
    
    groups = 2 if groups < 2 else groups
    bar_width = 1. / groups
    for weather_var in ["PRECIP", "TEMP"]:
        stationarity_fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
        stationarity_fig.suptitle("{} Residuals Stationarity Check with {} Groups | Above Dashed Line Implies Stationarity".format(weather_var.capitalize(), groups))
        stationarity_fig.supxlabel("Month"), stationarity_fig.supylabel("Mann-Whitney U p-Value [-]")
        months = list(pt_dict.keys())
        mwu_pvalues = np.full(shape=(len(months), groups-1), fill_value=np.nan)
        for m, month in enumerate(months):
            resids = pt_dict[month][weather_var + " ARFit"].resid
            group_chunk = len(resids)//groups if len(resids) % groups == 0 else len(resids)//groups+1
            group_data = []
            for n in range(groups):
                group_data.append(resids[n*group_chunk:(n+1)*group_chunk])
            for g in range(groups-1):
                x_data, y_data = np.array(group_data[g], dtype=float), np.array(group_data[g+1], dtype=float)
                mask = ~(np.isnan(x_data) | np.isnan(y_data))
                mwu_pvalues[m, g] = scipy.stats.mannwhitneyu(x=x_data[mask], y=y_data[mask])[1]

        # actually plotting
        axis.grid()
        axis.set(ylim=[0, 1])
        for g in range(groups-1):
            axis.bar([x+g*bar_width for x in range(len(months))], mwu_pvalues[:, g], width=bar_width, zorder=10)
        axis.hlines(0.05, -1, 12, color="black", linestyles="dashed", zorder=11)
        axis.set_xticks([m+(groups-2)*(bar_width/2) for m in range(len(months))])
        axis.set_xticklabels(labels=months, rotation=45)
        plt.tight_layout()
        stationarity_fig.savefig("{}Validate_{}_Resid_Stationarity_with{}Groups.svg".format(dp, weather_var, groups))
        plt.close()


def validate_pt_dependence_structure(dp: str, pt_dict: dict) -> None:
    """
    Validation figure for checking the dependence structure of 
    copula families for the precipitation and temperature residuals 
    through K-plots
    (Genest & Boies, 2003: https://www.jstor.org/stable/30037296) 

    Parameters
    ----------
    dp: str
        Filepath for saving the validation figure
    pt_dict: pd.DataFrame
        The precipitation and temperature data organized by month, as a dict
    """
    
    # define the functional form of the integrand for W_i:n
    def W_inIntegrand(w, idx, num):
        scale = num * math.comb(num - 1, idx - 1)
        u = w - (w * np.log(w))
        return scale * w * u ** (idx - 1) * (1 - u) ** (num - idx) * -np.log(w)

    kplots_fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9), sharex="all", sharey="all")
    kplots_fig.suptitle("Monthly K-Plots | Above Dashed Line: (+) Dependence, Below Dashed Line: (-) Dependence ")
    kplots_fig.supxlabel("$W_{i:n}$"), kplots_fig.supylabel("$H_{(i)}$")
    months = list(pt_dict.keys())
    for i, axis in enumerate(axes.flat):
        n = len(pt_dict[months[i]]["PRECIP ARFit"].resid)
        H_i, W_in = np.array([]), np.array([])

        # calculate W_in, H_i
        for ii in range(n):
            p_i = pt_dict[months[i]]["PRECIP ARFit"].resid[ii]
            T_i = pt_dict[months[i]]["TEMP ARFit"].resid[ii]
            if np.isnan(p_i) or np.isnan(T_i):
                continue
            H = (1 / (n - 1)) * sum([1 for j in range(n) if j != ii and
                                     pt_dict[months[i]]["PRECIP ARFit"].resid[j] <= p_i and
                                     pt_dict[months[i]]["TEMP ARFit"].resid[j] <= T_i])
            H_i = np.append(H_i, H)
            W_in = np.append(W_in, scipy.integrate.quad(W_inIntegrand, a=0, b=1, args=(ii + 1, n))[0])
        H_i.sort()

        # plot H_(i) against W_in
        axis.grid()
        axis.set_title(months[i])
        axis.plot([jj / n for jj in range(n)], [jj / n for jj in range(n)], c="black", linestyle="dashed")
        axis.plot(W_in, H_i, c="magenta")
    plt.tight_layout()
    kplots_fig.savefig("{}Validate_KPlots.svg".format(dp))
    plt.close()
