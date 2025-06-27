import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


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

