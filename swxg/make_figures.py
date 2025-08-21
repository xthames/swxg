import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib as mpl
from statsmodels.graphics.tsaplots import plot_acf
import scipy
import math
from multiprocessing import Process
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
import warnings
import datetime as dt


def squarest_subplots(n: int) -> tuple[int]:
    """
    For some input value, return the most rectangular (or square)
    2D arrangement of that input value

    Parameters
    ----------
    n: int
        Number of elements to rectangularize

    Returns
    -------
    rows: int
        Number of rows in the 2D arrangement
    cols: int
        Number of cols in the 2D arrangement
    """
    
    rows = int(n ** 0.5)
    cols = int(np.ceil(n / rows))
    return rows, cols


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
    num_states_fig.savefig("{}Validate_GMMHMM_NumStates.svg".format(dp))
    plt.close()


def validate_explore_pt_dependence(dp: str, pt_data: pd.DataFrame, good_years: list[int]) -> None:
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
    good_years: list[int]
        The list of years fit by the GMMHMM
    """
    
    sites = sorted(set(pt_data["SITE"].values))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # spatially-averaged correlation between precipitation and temperature
    spatial_corr_df = pd.DataFrame({"Kendall": np.nan, "Spearman": np.nan}, index=months)
    spatial_data_dict = {month: {"PRECIP": [], "TEMP": []} for month in months}
    for month in months:
        month_index = pt_data["MONTH"] == month
        for year in good_years:
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
    sa_corr_fig.savefig("{}Validate_Copulae_ExplorePTCorrelation_MonthlySpatialAverage.svg".format(dp))
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
    pt_dist_fig.savefig("{}Validate_Copulae_PTDistribution_MonthlySpatialAverage.svg".format(dp))
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
        acf_fig.savefig("{}Validate_Copulae_{}_ACF.svg".format(dp, weather_var.capitalize()))
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
                x_mask, y_mask = ~np.isnan(x_data), ~np.isnan(y_data)
                mwu_pvalues[m, g] = scipy.stats.mannwhitneyu(x=x_data[x_mask], y=y_data[y_mask])[1]

        # actually plotting
        axis.grid()
        axis.set(ylim=[0, 1])
        for g in range(groups-1):
            axis.bar([x+g*bar_width for x in range(len(months))], mwu_pvalues[:, g], width=bar_width, zorder=10)
        axis.hlines(0.05, -1, 12, color="black", linestyles="dashed", zorder=11)
        axis.set_xticks([m+(groups-2)*(bar_width/2) for m in range(len(months))])
        axis.set_xticklabels(labels=months, rotation=45)
        plt.tight_layout()
        stationarity_fig.savefig("{}Validate_Copulae_{}_ResidStationarity{}Groups.svg".format(dp, weather_var.title(), groups))
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
    pt_dict: dict
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
    kplots_fig.savefig("{}Validate_Copulae_KPlots.svg".format(dp))
    plt.close()


def validate_pt_fits(dp: str, data_df: pd.DataFrame, precip_dict: dict, temp_dict: dict) -> None:
    """
    Validation manager for all of the figures that can be generated
    after fitting precipitation and temperature

    Parameters
    ----------
    dp: str
        Filepath for saving the validation figure
    data_df: pd.DataFrame
        Temporally-formated precipitation and temperature data, as a dataframe
    precip_dict: 
        The fitted precipitation data, as a dict
    temp_dict:
        The fitted temperature and copulae data, as a dict
    """
    
    # helper function, parallelizing plotting
    def multiprocess_helper(fns, fninputs):
        pross = []
        for i, fn in enumerate(fns):
            p = Process(target=fn, args=fninputs[i])
            p.start()
            pross.append(p)
        for p in pross:
            p.join()
    
    # functions to call
    validate_obs_spatial_temporal_correlations(dp, data_df, precip_dict, temp_dict)
    validate_gmmhmm_statistics(dp, data_df, precip_dict)
    validate_copulae_statistics(dp, data_df, temp_dict)


def validate_obs_spatial_temporal_correlations(dp: str, data: pd.DataFrame, p_dict: dict, t_dict: dict) -> None:
    """
    Validation figures for all the observed precipitation and temperature 
    spatial correlations, using the Pearson method (since there's no 
    comparison between parameters); precipitation temporal (Markovian) 
    structure at annual and monthly levels

    Parameters
    ----------
    dp: str
        Filepath for saving the validation figure
    data: pd.DataFrame
        Temporally-formated precipitation and temperature data, as a dataframe
    p_dict: dict 
        The fitted precipitation data, as a dict
    t_dict: dict
        The fitted temperature and copulae data, as a dict
    """
    
    sites = sorted(set(data["SITE"].values))
    good_years = list(p_dict["log10_annual_precip"].index)
    years = [y for y in range(min(good_years), max(good_years)+1)]
    month_names = list(t_dict.keys())
    month_names_to_nums_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                                "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    empty_spatial_df = pd.DataFrame(index=years, columns=sites, dtype=float)
    spatial_dict = {"p": {k: empty_spatial_df.copy() for k in ["annual", *month_names]}, 
                    "T": {k: empty_spatial_df.copy() for k in ["annual", *month_names]}}
    temporal_dict = {"annual": {}, "monthly": {}}
    for site in sites:
        site_idx = data["SITE"] == site
        site_entry = data.loc[site_idx]
        for year in years:
            year_idx = site_entry["YEAR"] == year
            year_entry = site_entry.loc[year_idx]
            precip_annual = np.nan if len(year_entry["PRECIP"].values) == 0 else np.nansum(year_entry["PRECIP"].values)
            temp_annual = np.nan if len(year_entry["TEMP"].values) == 0 else np.nanmean(year_entry["TEMP"].values)
            # -- spatial
            spatial_dict["p"]["annual"].at[year, site] = precip_annual 
            spatial_dict["T"]["annual"].at[year, site] = temp_annual
            # -- temporal
            temporal_dict["annual"][(site, year)] = [site, year, np.log10(precip_annual)]
            for month in month_names:
                month_num = month_names_to_nums_dict[month]
                month_idx = year_entry["MONTH"] == month_num
                month_entry = year_entry.loc[month_idx]
                precip_monthly = np.nan if len(month_entry["PRECIP"].values) == 0 else np.nansum(month_entry["PRECIP"].values)
                temp_monthly = np.nan if len(month_entry["TEMP"].values) == 0 else np.nanmean(month_entry["TEMP"].values)
                # -- spatial
                spatial_dict["p"][month].at[year, site] = precip_monthly
                spatial_dict["T"][month].at[year, site] = temp_monthly
                #  -- temporal
                temporal_dict["monthly"][(site, year, month)] = [site, year, month, precip_monthly]
    temporal_dict["annual"] = pd.DataFrame().from_dict(temporal_dict["annual"], orient="index", columns=["SITE", "YEAR", "PRECIP"])
    temporal_dict["annual"].reset_index(drop=True, inplace=True)
    temporal_dict["annual"].astype({"SITE": str, "YEAR": int, "PRECIP": float})
    temporal_dict["monthly"] = pd.DataFrame().from_dict(temporal_dict["monthly"], orient="index", columns=["SITE", "YEAR", "MONTH", "PRECIP"])
    temporal_dict["monthly"].reset_index(drop=True, inplace=True)
    temporal_dict["monthly"].astype({"SITE": str, "YEAR": int, "MONTH": str, "PRECIP": float})

    # plot spatial
    corr_cmap = plt.get_cmap("gnuplot", 21)
    for k in spatial_dict.keys():
        wvar = "precip" if k == "p" else "temp"
        wvar_dict = spatial_dict[k]
        for kk in wvar_dict.keys():
            corr_df = wvar_dict[kk].corr(method="pearson")
            spatial_fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
            corr_colors = axis.imshow(corr_df.values, vmin=0, vmax=1, cmap=corr_cmap)
            spatial_fig.colorbar(corr_colors, label="{} {} Spatial Correlation [Pearson, -]".format(kk.title(), wvar.title()))
            plt.xticks(range(len(sites)), sites, rotation=75)
            plt.yticks(range(len(sites)), sites, rotation=0)
            spatial_fig.savefig("{}Validate_SpatialCorrelation_{}_{}.svg".format(dp, kk.title(), wvar.title()))
            plt.close()

    # plot temporal
    rs, cs = squarest_subplots(len(sites)) 
    acf_handle = mpl.lines.Line2D([], [], color="royalblue", marker="o")
    racf_handle = mpl.lines.Line2D([], [], color="rebeccapurple", marker="o")
    pacf_handle = mpl.lines.Line2D([], [], color="chocolate", marker="o")
    for time_scale in ["annual", "monthly"]:
        temporal_fig, axes = plt.subplots(nrows=rs, ncols=cs, figsize=(16, 9), sharex="all", sharey="all")
        temporal_fig.suptitle("ACF and PACF for {} Precip".format(time_scale.title()))
        temporal_fig.supxlabel("Lag"), temporal_fig.supylabel("Index Value [-]")
        for a, axis in enumerate(axes.flat):
            axis.grid()
            site = sites[a]
            site_idx = temporal_dict[time_scale]["SITE"] == site
            site_entry = temporal_dict[time_scale].loc[site_idx]
            precip_data = site_entry["PRECIP"].values
            precip_data = precip_data[~np.isnan(precip_data)]
            raw_acf = plot_acf(ax=axis, x=precip_data, use_vlines=False, color="royalblue")
            raw_ar1fit = AutoReg(precip_data, lags=[1]).fit()
            resid_acf = plot_acf(ax=axis, x=raw_ar1fit.resid, use_vlines=False, color="rebeccapurple")
            raw_pacf = plot_pacf(ax=axis, x=precip_data, method="ywm", use_vlines=False, color="chocolate")
            if a == 0: axis.legend(handles=[acf_handle, racf_handle, pacf_handle], labels=["Obs ACF", "Obs AR(1) Resid ACF", "Obs PACF"])
            axis.set(title=site, xlim=(-0.25, 11.25))
            axis.hlines(0, xmin=0, xmax=11, linestyles="dashed", color="black")
        plt.tight_layout()
        temporal_fig.savefig("{}Validate_GMMHMM_MarkovianStructure_{}.svg".format(dp, time_scale.title()))
        plt.close()


def validate_gmmhmm_statistics(dp: str, data: pd.DataFrame, p_dict: dict) -> None:
    """
    Validation figures for the precipitation GMMHMM, confirming:
    * that the transition between hidden states is Markovian (if 
      more than one state is found)
    * the solved hidden state as a function of date (year)
    * the transition probabilities between states are sensible
    * that the transformed precipitation data is Gaussian
    
    Parameters
    ----------
    dp: str
        Filepath for saving the validation figure
    data: pd.DataFrame
        Temporally-formated precipitation and temperature data, as a dataframe
    p_dict: dict
        The fitted precipitation data, as a dict
    """

    sites = sorted(set(data["SITE"].values))
    good_years = list(p_dict["log10_annual_precip"].index)
    years = [y for y in range(min(good_years), max(good_years)+1)]
    rs, cs = squarest_subplots(len(sites)) 
    state_colors = ["blue", "green", "cyan", "magenta", "yellow"]

    # Q-Q plots -- are the log10-transformed annual precipitation data normal (they should be) 
    # --> confirming "GMM" in GMMHMM 
    qq_fig, axes = plt.subplots(nrows=rs, ncols=cs, figsize=(16, 9))
    qq_fig.suptitle("Q-Q Plot of States' Log-Normal Distributions vs. Annual Data")
    qq_fig.supxlabel("Theoretical Quantiles [-]"), qq_fig.supylabel("Data Quantiles [-]")
    for a, axis in enumerate(axes.flat):
        axis.grid()
        axis.set(title=sites[a])
        mus = [p_dict["means"][s][a] for s in range(p_dict["num_gmmhmm_states"])]
        stds = [p_dict["stds"][s][a] for s in range(p_dict["num_gmmhmm_states"])]
        state_leg_strs = ["State {}".format(s+1) for s in range(p_dict["num_gmmhmm_states"])]
        for s in range(p_dict["num_gmmhmm_states"]):
            state_idx = p_dict["hidden_states"] == s
            data = p_dict["log10_annual_precip"][sites[a]].values[state_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                sm.qqplot(ax=axis, data=data, dist=scipy.stats.norm, loc=mus[s], scale=stds[s], line="45", 
                          **{"markerfacecolor": mpl.colors.to_rgba(state_colors[s], 0.67), 
                             "markeredgecolor": mpl.colors.to_rgba(state_colors[s], 0.67)})
        if a == 0: axis.legend(state_leg_strs)
        axis.set_xlabel(""), axis.set_ylabel("")
    plt.tight_layout()
    qq_fig.savefig("{}Validate_GMMHMM_QQs.svg".format(dp))
    plt.close()
    
    # ACF and PACF plots -- are the hidden states Markovian (they should be)
    # --> confirming "HMM" in GMMHMM  
    if p_dict["num_gmmhmm_states"] > 1:
        hiddenstate_markov_fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 9), sharex="all")
        hiddenstate_markov_fig.suptitle("Markovian Structure of GMMHMM Hidden States")
        hiddenstate_markov_fig.supxlabel("Lag")
        for i, axis in enumerate(axes.flat):
            axis.grid()
            if i == 0:
                plot_acf(ax=axis, x=p_dict["hidden_states"], vlines_kwargs={"color": "black"})
                axis.set(ylabel="ACF [-]", xlim=(-0.25, 16.25))
                axis.hlines(0, xmin=0, xmax=24, color="red")
            else:
                plot_pacf(ax=axis, x=p_dict["hidden_states"], method="ywm", vlines_kwargs={"color": "black"})
                axis.set(ylabel="PACF [-]", xlim=(-0.25, 16.25))
                axis.hlines(0, xmin=0, xmax=24, color="red")
        plt.tight_layout()
        hiddenstate_markov_fig.savefig("{}Validate_GMMHMM_HiddenStateMarkovStructure.svg".format(dp))
        plt.close()

    # plot the transition probability matrix 
    tprob_fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), sharex="all", sharey="all")
    tprob_fig.suptitle("Transition Probabilities from GMMHMM")
    tprob_im = axes.matshow(p_dict["t_probs"], vmin=0, vmax=1, cmap="gnuplot")
    for i in range(p_dict["t_probs"].shape[0]):
        for j in range(p_dict["t_probs"].shape[1]):
            axes.text(j, i, '{:.3f}'.format(p_dict["t_probs"][i, j]), ha='center', va='center', backgroundcolor=[0, 0, 0, 0.25], color="white")
    axes.set(xticks=range(p_dict["num_gmmhmm_states"]), xticklabels=["To State {}".format(s) for s in range(p_dict["num_gmmhmm_states"])],
             yticks=range(p_dict["num_gmmhmm_states"]), yticklabels=["From State {}".format(s) for s in range(p_dict["num_gmmhmm_states"])])
    tprob_fig.colorbar(tprob_im, label="Probability [-]")
    plt.tight_layout()
    tprob_fig.savefig("{}Validate_GMMHMM_TransitionProbabilities.svg".format(dp))
    plt.close()


def validate_copulae_statistics(dp: str, data: pd.DataFrame, t_dict: dict) -> None:
    """
    Validation figures for the copulae, confirming:
    * best-fitting copulae via AIC comparison
    * copulae plots vs. empirical copula
    
    Parameters
    ----------
    dp: str
        Filepath for saving the validation figure
    data: pd.DataFrame
        Temporally-formated precipitation and temperature data, as a dataframe
    t_dict: dict 
        The fitted precip/copulae/temperature data, as a dict
    """
    
    # aics
    aic_fig = plt.figure(figsize=(16, 9))
    aic_fig.suptitle('Monthly AIC per Copula')
    months = list(t_dict.keys())
    ind_aic, frk_aic, gau_aic = [np.nan] * len(months), [np.nan] * len(months), [np.nan] * len(months)
    for j, month in enumerate(months):
        fit_info = t_dict[month]["CopulaDF"]
        for idx in fit_info.index:
            if idx == "Independence": ind_aic[j] = fit_info.at[idx, "AIC"]
            if idx == "Frank": frk_aic[j] = fit_info.at[idx, "AIC"]
            if idx == "Gaussian": gau_aic[j] = fit_info.at[idx, "AIC"]
    ind_aic = [*ind_aic, ind_aic[0]]
    frk_aic = [*frk_aic, frk_aic[0]]
    gau_aic = [*gau_aic, gau_aic[0]]
    months = [*months, months[0]]
    month_label_loc = np.linspace(0, 2 * np.pi, num=len(months))
    axis = aic_fig.add_subplot(1, 1, 1, projection="polar")
    axis.spines["polar"].set_visible(False)
    if not np.all(np.isnan(ind_aic)): axis.plot(month_label_loc, ind_aic, label="Ind.", c="grey")
    if not np.all(np.isnan(frk_aic)): axis.plot(month_label_loc, frk_aic, label="Frk.", c="blue")
    if not np.all(np.isnan(gau_aic)): axis.plot(month_label_loc, gau_aic, label="Gau.", c="red")
    axis.set_thetagrids(np.degrees(month_label_loc), labels=months)
    axis.legend(loc="upper right")
    plt.tight_layout()
    aic_fig.savefig("{}Validate_Copulae_AICs.svg".format(dp))
    plt.close()
 
    # copula comparison
    def calculate_empirical_copula(pseudo_observations, resolution=50):
        n_data, n_dim = pseudo_observations.shape[0], pseudo_observations.shape[1]
        u0 = np.linspace(0, 1, resolution)
        Cn = np.full(shape=(resolution,)*n_dim, fill_value=0.)
        for k in range(n_data):
            U1, U2 = pseudo_observations[k, 0], pseudo_observations[k, 1]
            for ii in range(resolution):
                for jj in range(resolution):
                    if U1 <= u0[ii] and U2 <= u0[jj]:
                        Cn[ii, jj] += 1.
        Cn /= n_data
        return Cn, u0
    def approximate_theoretical_copula(copula, marginals, use_df=False):
        C_theory = np.full(shape=(len(marginals), len(marginals)), fill_value=np.nan)
        for ii, u1 in enumerate(marginals):
            for jj, u2 in enumerate(marginals):
                data_df = pd.DataFrame(data={"uP": [u1], "uT": [u2]}, dtype=float)
                C_val = copula.cdf(data_df.values) if use_df else copula.cdf(data_df.values)
                C_theory[ii, jj] = C_val[0] if type(C_val) in [list, np.ndarray] else C_val
        return C_theory 
    copulae_fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9), sharex="all", sharey="all")
    copulae_fig.suptitle("Copulae Comparison")
    copulae_fig.supxlabel("$U_P$"), copulae_fig.supylabel("$U_T$")
    legend_symbols, legend_labels = [mpl.patches.Patch(color="black")], ["Emp."]
    for i, axis in enumerate(axes.flat):
        month = months[i]
        p_obs = np.array([t_dict[month]["PRECIP pObs"], t_dict[month]["TEMP pObs"]]).T
        emp_C, u = calculate_empirical_copula(p_obs)
        axis.grid()
        axis.set_title(month)
        contour_levels = [(i + 1) / 10 for i in range(10)]
        eContour = axis.contour(u, u, emp_C, levels=contour_levels, colors="black", vmin=0, vmax=1, linewidths=0.75)
        axis.clabel(eContour, inline=True, fontsize=10)
        copula_fit_df = t_dict[month]["CopulaDF"]
        for family in copula_fit_df.index.values:
            if family == "Independence":
                axis.contour(u, u, approximate_theoretical_copula(copula_fit_df.at[family, "Copula"], u),
                             levels=contour_levels, colors="grey", vmin=0, vmax=1, linewidths=0.75, linestyles="dashed")
                if i == 8: legend_symbols.append(mpl.patches.Patch(color="grey")), legend_labels.append("Ind.")
            if family == "Frank":
                axis.contour(u, u, approximate_theoretical_copula(copula_fit_df.at[family, "Copula"], u),
                             levels=contour_levels, colors="blue", vmin=0, vmax=1, linewidths=0.75, linestyles="dashed")
                if i == 8: legend_symbols.append(mpl.patches.Patch(color="blue")), legend_labels.append("Frk.")
            if family == "Gaussian":
                axis.contour(u, u, approximate_theoretical_copula(copula_fit_df.at[family, "Copula"], u, use_df=True),
                             levels=contour_levels, colors="red", vmin=0, vmax=1, linewidths=0.75, linestyles="dashed")
                if i == 8: legend_symbols.append(mpl.patches.Patch(color="red")), legend_labels.append("Gau.")
        if i == 8: axis.legend(handles=legend_symbols, labels=legend_labels, loc="lower left")
        axis.scatter(p_obs[:, 0], p_obs[:, 1], c="black", s=1)
    plt.tight_layout()
    copulae_fig.savefig("{}Validate_Copulae_Comparison.svg".format(dp))
    plt.close()


def compare_synth_to_obs(dp: str, synth_df: pd.DataFrame, obs_df: pd.DataFrame) -> None:
    """
    Comparing the synthesized WX data with the observed WX data through
    a variety of statistical and visual tests

    Parameters
    ----------
    dp: str
        Filepath for saving the validation figure
    synth_df: pd.DataFrame
        Synthesized precipitation and temperature data
    obs_df: pd.DataFrame 
        Observed precipitation and temperature data
    """
    
    def agg_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
        """
        Take a daily resolution dataframe and convert to monthly

        Parameters
        ----------
        df: pd.DataFrame
            Daily dataframe to transform
        
        Returns
        -------
        monthly_df: pd.DataFrame
            Monthly-transformed dataframe
        """
        
        df_dict = {}
        for site in sorted(set(df["SITE"].values)):
            obs_site_idx = df["SITE"] == site
            obs_site_entry = df.loc[obs_site_idx]
            for year in sorted(set(obs_site_entry["YEAR"].values)):
                obs_year_idx = obs_site_entry["YEAR"] == year
                obs_year_entry = obs_site_entry.loc[obs_year_idx]
                for month in sorted(set(obs_year_entry["MONTH"].values)):
                    obs_month_idx = obs_year_entry["MONTH"] == month
                    obs_month_entry = obs_year_entry.loc[obs_month_idx]
                    prcps = obs_month_entry["PRECIP"].values
                    temps = obs_month_entry["TEMP"].values
                    prcp = np.nan if all(np.isnan(prcps)) or len(prcps) == 0 else np.nansum(prcps)
                    temp = np.nan if all(np.isnan(temps)) or len(temps) == 0 else np.nanmean(temps)
                    df_dict[(site, year, month)] = [site, year, month, prcp, temp]
        monthly_df = pd.DataFrame().from_dict(df_dict, orient="index", columns=["SITE", "YEAR", "MONTH", "PRECIP", "TEMP"])
        monthly_df.reset_index(drop=True, inplace=True)
        monthly_df.astype({"SITE": str, "YEAR": int, "MONTH": int, "PRECIP": float, "TEMP": float})
        return monthly_df

    sites = sorted(set(obs_df["SITE"].values))

    # annual precip histograms
    r_sites, c_sites = squarest_subplots(len(set(obs_df["SITE"].values)))
    comp_gmmhmm_fig, axes = plt.subplots(nrows=r_sites, ncols=c_sites, figsize=(16, 9), sharex="all", sharey="all")
    comp_gmmhmm_fig.suptitle("Comparison of Obs and Synth Annual Precipitation via Histogram")
    comp_gmmhmm_fig.supxlabel("Total Annual Precipitation [m]"), comp_gmmhmm_fig.supylabel("Probability Density [-]")
    for a, axis in enumerate(axes.flat):
        if a >= len(sites): continue
        obs_site_idx, synth_site_idx = obs_df["SITE"] == sites[a], synth_df["SITE"] == sites[a]
        obs_site_entry, synth_site_entry = obs_df.loc[obs_site_idx], synth_df.loc[synth_site_idx]
        obs_annuals, synth_annuals = [], []
        for year in sorted(set(obs_site_entry["YEAR"].values)):
            obs_annuals.append(np.nansum(obs_site_entry.loc[obs_site_entry["YEAR"] == year, "PRECIP"].values))
        for year in sorted(set(synth_site_entry["YEAR"].values)):
            synth_annuals.append(np.nansum(synth_site_entry.loc[synth_site_entry["YEAR"] == year, "PRECIP"].values))
        logbins = np.logspace(np.nanmin(np.log10(obs_annuals)), np.nanmax(np.log10(obs_annuals)), 10)
        axis.set(title=sites[a])
        axis.set(xscale="log")
        axis.hist(synth_annuals, density=True, bins=logbins, color="grey", alpha=1)
        axis.hist(obs_annuals, density=True, bins=logbins, color="black", histtype="step")
        if a == 0: axis.legend(["Obs", "Synth"])
    plt.tight_layout()
    comp_gmmhmm_fig.savefig("{}Compare_GMMHMM_AnnualPrecip.svg".format(dp))
    plt.close()
     
    # cumulative frequency of precipitation plot
    comp_cumfreq_fig, axes = plt.subplots(nrows=r_sites, ncols=c_sites, figsize=(16, 9), sharex="all", sharey="all")
    comp_cumfreq_fig.suptitle("Cumulative Frequency Curves for Precipitation")
    comp_cumfreq_fig.supxlabel("% Exceedance")
    comp_cumfreq_fig.supylabel("Daily Precipitation [m]" if "DAY" in obs_df.columns else "Monthly Precipitation [m]")
    for a, axis in enumerate(axes.flat):
        axis.set(title=sites[a])
        axis.set(yscale="log", ylim=[1E-5, 1E-1]) if "DAY" in obs_df.columns else axis.set(yscale="log", ylim=[2.54E-4, 1])
        obs_site_idx, synth_site_idx = obs_df["SITE"] == sites[a], synth_df["SITE"] == sites[a]
        obs_site_entry, synth_site_entry = obs_df.loc[obs_site_idx], synth_df.loc[synth_site_idx]
        descending_obs = np.array(sorted(obs_site_entry["PRECIP"].values, reverse=True))
        descending_synth = np.array(sorted(synth_site_entry["PRECIP"].values, reverse=True))
        obs_exceedance = np.array([np.nansum(descending_obs >= point) / len(descending_obs) for point in descending_obs])
        synth_exceedance = np.array([np.nansum(descending_synth >= point) / len(descending_synth) for point in descending_synth])
        axis.plot(100 * obs_exceedance, descending_obs, c="black", label="Obs")
        axis.plot(100 * synth_exceedance, descending_synth, c="grey", linestyle="-.", label="Synth")
        if a == 0: axis.legend()
    plt.tight_layout()
    comp_cumfreq_fig.savefig("{}Compare_CumulativeFrequency_Precip.svg".format(dp))
    plt.close()
    
    # monthly spatial correlations for precip and temp
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if "DAY" in obs_df.columns:
        obs_monthly_df, synth_monthly_df = agg_to_monthly(obs_df), agg_to_monthly(synth_df)
    else:
        obs_monthly_df, synth_monthly_df = obs_df, synth_df
    obs_monthly_dict, synth_monthly_dict = {}, {}
    for month in sorted(set(obs_monthly_df["MONTH"].values)):
        obs_monthly_dict[int(month)] = {"PRECIP": pd.DataFrame(), "TEMP": pd.DataFrame()}
        obs_month_idx = obs_monthly_df["MONTH"] == month
        obs_month_entry = obs_monthly_df.loc[obs_month_idx]
        for site in sorted(set(obs_month_entry["SITE"].values)):
            obs_site_idx = obs_month_entry["SITE"] == site
            obs_site_entry = obs_month_entry.loc[obs_site_idx]
            obs_monthly_dict[month]["PRECIP"][site] = obs_site_entry["PRECIP"].values
            obs_monthly_dict[month]["TEMP"][site] = obs_site_entry["TEMP"].values
    for month in sorted(set(synth_monthly_df["MONTH"].values)):
        synth_monthly_dict[int(month)] = {"PRECIP": pd.DataFrame(), "TEMP": pd.DataFrame()}
        synth_month_idx = synth_monthly_df["MONTH"] == month
        synth_month_entry = synth_monthly_df.loc[synth_month_idx]
        for site in sorted(set(synth_month_entry["SITE"].values)):
            synth_site_idx = synth_month_entry["SITE"] == site
            synth_site_entry = synth_month_entry.loc[synth_site_idx]
            synth_monthly_dict[month]["PRECIP"][site] = synth_site_entry["PRECIP"].values
            synth_monthly_dict[month]["TEMP"][site] = synth_site_entry["TEMP"].values
    corr_cmap = plt.get_cmap("gnuplot", 21)
    for month in obs_monthly_dict.keys():
        compare_spatial_fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), sharex="all", sharey="all")
        compare_spatial_fig.suptitle("Compare Spatial Correlations: {}".format(month_names[month-1]))
        for i, wvar in enumerate(["PRECIP", "TEMP"]):
            obs_axis, synth_axis = axes[i, 0], axes[i, 1]
            obs_axis.set_title("Obs {}".format(wvar.title())), synth_axis.set_title("Synth {}".format(wvar.title()))
            obs_corr = obs_monthly_dict[month][wvar].corr(method="pearson") 
            synth_corr = synth_monthly_dict[month][wvar].corr(method="pearson")
            obs_axis.imshow(obs_corr.values, vmin=0, vmax=1, cmap=corr_cmap)
            corr_colors = synth_axis.imshow(synth_corr.values, vmin=0, vmax=1, cmap=corr_cmap)
            obs_axis.set_xticks(range(len(sites))), obs_axis.set_yticks(range(len(sites)))
            obs_axis.set_xticklabels(sites, rotation=45, ha="right"), obs_axis.set_yticklabels(sites)
            synth_axis.set_xticks(range(len(sites))), synth_axis.set_yticks(range(len(sites)))
            synth_axis.set_xticklabels(sites, rotation=45, ha="right"), synth_axis.set_yticklabels(sites)
        compare_spatial_fig.colorbar(corr_colors, ax=axes[:, 1], label="Pearson Correlation Coeff [-]")
        compare_spatial_fig.savefig("{}Compare_SpatialCorrelations_{}.svg".format(dp, month_names[month-1]))
        plt.close()
        
    # monthly temporal correlations for precip and temp
    for s, site in enumerate(sites):
        compare_temporal_fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(9, 16), sharex="all", sharey="row")
        compare_temporal_fig.suptitle("Compare Monthly Temporal Correlations: {}".format(site))
        compare_temporal_fig.supxlabel("Lag")
        obs_precip_acf_axis, synth_precip_acf_axis = axes[0, 0], axes[0, 1]
        obs_precip_pacf_axis, synth_precip_pacf_axis = axes[1, 0], axes[1, 1]
        obs_temp_acf_axis, synth_temp_acf_axis = axes[2, 0], axes[2, 1]
        obs_temp_pacf_axis, synth_temp_pacf_axis = axes[3, 0], axes[3, 1]
        obs_site_idx, synth_site_idx = obs_monthly_df["SITE"] == site, synth_monthly_df["SITE"] == site
        obs_site_entry, synth_site_entry = obs_monthly_df.loc[obs_site_idx], synth_monthly_df.loc[synth_site_idx]
        plot_acf(ax=obs_precip_acf_axis, x=obs_site_entry["PRECIP"].values, use_vlines=False, color="royalblue")
        plot_acf(ax=synth_precip_acf_axis, x=synth_site_entry["PRECIP"].values, use_vlines=False, color="royalblue")
        plot_pacf(ax=obs_precip_pacf_axis, x=obs_site_entry["PRECIP"].values, use_vlines=False, color="royalblue")
        plot_pacf(ax=synth_precip_pacf_axis, x=synth_site_entry["PRECIP"].values, use_vlines=False, color="royalblue")
        plot_acf(ax=obs_temp_acf_axis, x=obs_site_entry["TEMP"].values, use_vlines=False, color="firebrick")
        plot_acf(ax=synth_temp_acf_axis, x=synth_site_entry["TEMP"].values, use_vlines=False, color="firebrick")
        plot_pacf(ax=obs_temp_pacf_axis, x=obs_site_entry["TEMP"].values, use_vlines=False, color="firebrick")
        plot_pacf(ax=synth_temp_pacf_axis, x=synth_site_entry["TEMP"].values, use_vlines=False, color="firebrick")
        obs_precip_acf_axis.set_title("Obs"), synth_precip_acf_axis.set_title("Synth")
        for axis in [obs_precip_pacf_axis, synth_precip_pacf_axis, obs_temp_acf_axis, synth_temp_acf_axis, obs_temp_pacf_axis, synth_temp_pacf_axis]:
            axis.set_title("")
        obs_precip_acf_axis.set_ylabel("Precip ACF [-]"), obs_temp_acf_axis.set_ylabel("Temp ACF [-]")
        obs_precip_pacf_axis.set_ylabel("Precip PACF [-]"), obs_temp_pacf_axis.set_ylabel("Temp PACF [-]")
        obs_temp_pacf_axis.set_xlim([0, 24])
        plt.tight_layout()
        compare_temporal_fig.savefig("{}Compare_TemporalCorrelations_{}.svg".format(dp, site.replace(" ", "")))
        plt.close()

    # compare the observed and synthetic Kendall/Spearman correlation coefficients
    compare_ks_fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), sharex="all", sharey="all")
    compare_ks_fig.suptitle("Correlation of Spatially Averaged P/T Data by Month")
    compare_ks_fig.supxlabel("Month"), compare_ks_fig.supylabel("Correlation Coefficient [-]") 
    obs_ks_corrs_dict, synth_ks_corrs_dict = {"Kendall": [], "Spearman": []}, {"Kendall": [], "Spearman": []}
    for month in obs_monthly_dict.keys():
        obs_month_idx, synth_month_idx = obs_monthly_df["MONTH"] == month, synth_monthly_df["MONTH"] == month
        obs_month_entry, synth_month_entry = obs_monthly_df.loc[obs_month_idx], synth_monthly_df.loc[synth_month_idx]
        obs_ks_dict = {"PRECIP": [], "TEMP": []}
        synth_ks_dict = {"PRECIP": [], "TEMP": []}
        for year in sorted(set(obs_month_entry["YEAR"].values)):
            obs_year_idx = obs_month_entry["YEAR"] == year
            obs_year_entry = obs_month_entry.loc[obs_year_idx]
            obs_ks_dict["PRECIP"].append(np.nanmean(obs_year_entry["PRECIP"].values))
            obs_ks_dict["TEMP"].append(np.nanmean(obs_year_entry["TEMP"].values))
        for year in sorted(set(synth_month_entry["YEAR"].values)):
            synth_year_idx = synth_month_entry["YEAR"] == year
            synth_year_entry = synth_month_entry.loc[synth_year_idx]
            synth_ks_dict["PRECIP"].append(np.nanmean(synth_year_entry["PRECIP"].values))
            synth_ks_dict["TEMP"].append(np.nanmean(synth_year_entry["TEMP"].values))
        obs_ks_df = pd.DataFrame({"P": obs_ks_dict["PRECIP"], "T": obs_ks_dict["TEMP"]}) 
        synth_ks_df = pd.DataFrame({"P": synth_ks_dict["PRECIP"], "T": synth_ks_dict["TEMP"]}) 
        obs_ks_corrs_dict["Kendall"].append(obs_ks_df.corr(method="kendall")["P"]["T"])
        obs_ks_corrs_dict["Spearman"].append(obs_ks_df.corr(method="spearman")["P"]["T"])
        synth_ks_corrs_dict["Kendall"].append(synth_ks_df.corr(method="kendall")["P"]["T"])
        synth_ks_corrs_dict["Spearman"].append(synth_ks_df.corr(method="spearman")["P"]["T"])
    for a, axis in enumerate(axes.flat):
        if a == 0: 
            ks_corrs_dict = obs_ks_corrs_dict
            axis.set_title("Obs")
            ks_color = "black"
        if a == 1:
            ks_corrs_dict = synth_ks_corrs_dict
            axis.set_title("Synth")
            ks_color = "grey"
        axis.plot(range(len(obs_monthly_dict.keys())), ks_corrs_dict["Kendall"], color=ks_color, marker="s", label=r"Kendall $\tau$", zorder=10)
        axis.plot(range(len(obs_monthly_dict.keys())), ks_corrs_dict["Spearman"], color=ks_color, marker="d", label=r"Spearman $\rho$", zorder=10)
        axis.legend()
        axis.hlines(0, 0, len(month_names)-1, color="red", linestyle="dashed", zorder=9) 
        axis.set_ylim(-1, 1)
        axis.set_xticks(range(len(month_names)))
        axis.set_xticklabels(month_names, rotation=45)
    plt.tight_layout()
    compare_ks_fig.savefig("{}Compare_PTCorrelations_KendallSpearman.svg".format(dp))
    plt.close()

    # compare via scatterplot and histogram the monthly observed and synthetic
    for site in sites:
        obs_site_idx, synth_site_idx = obs_monthly_df["SITE"] == site, synth_monthly_df["SITE"] == site
        compare_histscatter_fig = plt.figure(figsize=(16, 9))
        compare_histscatter_fig.suptitle("Direct Comparison of Precipitation and Temperature at {} | PRECIP on X, TEMP on Y".format(site))
        sub_figs = compare_histscatter_fig.subfigures(3, 4)
        for i, sub_fig in enumerate(sub_figs.flat):
            obs_month_idx, synth_month_idx = obs_monthly_df["MONTH"] == i+1, synth_monthly_df["MONTH"] == i+1
            axes = sub_fig.subplots(2, 2, gridspec_kw={"width_ratios": [4, 1], "height_ratios": [1, 3]})
            sub_fig.subplots_adjust(wspace=0, hspace=0)
            synthColor = "grey"
            for j, axis in enumerate(axes.flat):
                if j == 0:
                    # precip histogram comparison
                    axis.hist(synth_monthly_df.loc[synth_site_idx & synth_month_idx, "PRECIP"].values, density=True, color="grey")
                    axis.hist(obs_monthly_df.loc[obs_site_idx & obs_month_idx, "PRECIP"].values, density=True, color="black", histtype="step")
                    axis.set(xticks=[], yticks=[])
                if j == 1:
                    # label for the month
                    axis.axis("off")
                    axis.text(0.5, 0.5, month_names[i], transform=axis.transAxes, va="center", ha="center")
                if j == 2:
                    # scatterplot
                    axis.scatter(synth_monthly_df.loc[synth_site_idx & synth_month_idx, "PRECIP"].values,
                                 synth_monthly_df.loc[synth_site_idx & synth_month_idx, "TEMP"].values,
                                 marker=".", facecolors="grey", alpha=0.2, rasterized=True)
                    axis.scatter(obs_monthly_df.loc[obs_site_idx & obs_month_idx, "PRECIP"].values, 
                                 obs_monthly_df.loc[obs_site_idx & obs_month_idx, "TEMP"].values, 
                                 marker="o", facecolors="none", edgecolors="black")
                if j == 3:
                    # temp histogram comparison
                    axis.hist(synth_monthly_df.loc[synth_site_idx & synth_month_idx, "TEMP"].values, density=True, color="grey", orientation="horizontal")
                    axis.hist(obs_monthly_df.loc[obs_site_idx & obs_month_idx, "TEMP"].values, density=True, color="black", histtype="step", orientation="horizontal")
                    axis.set(xticks=[], yticks=[])
        compare_histscatter_fig.savefig("{}Compare_HistScatter_{}.svg".format(dp, site.replace(" ", "")))
        plt.close()

    # statistical tests for the observed/synthetic precipitation/temperature distributions
    # -- Mann-Whitney U (MWU), --> median
    # -- Levene --> std
    # -- Kolmogorov-Smirnov (T_n) --> empirical distribution sensitive to mean
    goodness_stats, bar_width = ["MWU", "Levene", "$T_n$"], 0.5
    r_months, c_months = squarest_subplots(len(month_names))
    for site in sites:
        obs_site_idx, synth_site_idx = obs_monthly_df["SITE"] == site, synth_monthly_df["SITE"] == site
        stats_fig, axes = plt.subplots(nrows=r_months, ncols=c_months, figsize=(16, 9), sharex="all", sharey="all")
        stats_fig.suptitle("Null-Hypothesis Statistics at {} | Mann-Whitney U, Levene, Kolmogorov-Smirnov".format(site))
        stats_fig.supylabel("p-Value [-]")
        for i, axis in enumerate(axes.flat):
            obs_month_idx, synth_month_idx = obs_monthly_df["MONTH"] == i+1, synth_monthly_df["MONTH"] == i+1
            obs_precips = obs_monthly_df.loc[obs_site_idx & obs_month_idx, "PRECIP"].astype(float).values 
            obs_temps = obs_monthly_df.loc[obs_site_idx & obs_month_idx, "TEMP"].astype(float).values
            synth_precips = synth_monthly_df.loc[synth_site_idx & synth_month_idx, "PRECIP"].astype(float).values 
            synth_temps = synth_monthly_df.loc[synth_site_idx & synth_month_idx, "TEMP"].astype(float).values
            prcpMWU, tavgMWU = scipy.stats.mannwhitneyu(obs_precips, synth_precips, nan_policy="omit"), scipy.stats.mannwhitneyu(obs_temps, synth_temps, nan_policy="omit")
            prcpLevene, tavgLevene = scipy.stats.levene(obs_precips, synth_precips[np.isfinite(synth_precips)]), scipy.stats.levene(obs_temps, synth_temps[np.isfinite(synth_temps)])
            prcpKS, tavgKS = scipy.stats.ks_2samp(obs_precips, synth_precips[np.isfinite(synth_precips)]), scipy.stats.ks_2samp(obs_temps, synth_temps[np.isfinite(synth_temps)])
            axis.set(title=month_names[i], xlim=[-bar_width * 1.5, 2 * (len(goodness_stats) - 1 + bar_width)], ylim=[0, 1])
            axis.bar([2 * x - bar_width / 2 for x in range(len(goodness_stats))], [prcpMWU[1], prcpLevene[1], prcpKS.pvalue], 
                     width=bar_width, color="royalblue", zorder=10, label="Precip")
            axis.bar([2 * x + bar_width / 2 for x in range(len(goodness_stats))], [tavgMWU[1], tavgLevene[1], tavgKS.pvalue], 
                     width=bar_width, color="orangered", zorder=10, label="Temp")
            if i == 11:
                axis.legend().set_zorder(11)
            axis.hlines(0.05, -2 * bar_width, 2 * (len(goodness_stats) + bar_width), color="black", linestyles="dashed", zorder=11)
            axis.set_xticks([2 * x for x in range(len(goodness_stats))])
            axis.set_xticklabels(labels=goodness_stats)
        plt.tight_layout()
        stats_fig.savefig("{}Compare_StatisticalDistributions_{}.svg".format(dp, site.replace(" ", "")))
        plt.close()

    # if daily, plot distribution by DOY
    if "DAY" in obs_df.columns:
        for site in sites:
            obs_site_idx, synth_site_idx = obs_df["SITE"] == site, synth_df["SITE"] == site
            obs_site_entry, synth_site_entry = obs_df.loc[obs_site_idx], synth_df.loc[synth_site_idx]
            doy_fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 9), sharex="all")
            doy_fig.suptitle("Comparison of Obs to Synth by DOY, with Medians (black, grey lines) and Interquartile Ranges (color, grey bands)")
            doy_fig.supxlabel("DOY")
            obs_doy_dict, synth_doy_dict = {}, {}
            for i in range(obs_site_entry.shape[0]):
                obs_i_entry = obs_site_entry.iloc[i]
                doy = int(dt.datetime.strptime("{}-{}-{}".format(str(obs_i_entry["YEAR"]).zfill(4), str(obs_i_entry["MONTH"]).zfill(2), str(obs_i_entry["DAY"]).zfill(2)), 
                                               "%Y-%m-%d").strftime("%j"))
                if doy not in obs_doy_dict:
                    obs_doy_dict[doy] = {"PRECIP": [obs_i_entry["PRECIP"]], "TEMP": [obs_i_entry["TEMP"]]}
                else:
                    obs_doy_dict[doy]["PRECIP"].append(obs_i_entry["PRECIP"])
                    obs_doy_dict[doy]["TEMP"].append(obs_i_entry["TEMP"])
            for i in range(synth_site_entry.shape[0]):
                synth_i_entry = synth_site_entry.iloc[i]
                if synth_i_entry["MONTH"] == 2 and synth_i_entry["DAY"] == 29: continue
                doy = int(dt.datetime.strptime("{}-{}-{}".format(str(synth_i_entry["YEAR"]).zfill(4), str(synth_i_entry["MONTH"]).zfill(2), str(synth_i_entry["DAY"]).zfill(2)), 
                                               "%Y-%m-%d").strftime("%j"))
                if doy not in synth_doy_dict:
                    synth_doy_dict[doy] = {"PRECIP": [float(synth_i_entry["PRECIP"])], "TEMP": [float(synth_i_entry["TEMP"])]}
                else:
                    synth_doy_dict[doy]["PRECIP"].append(float(synth_i_entry["PRECIP"]))
                    synth_doy_dict[doy]["TEMP"].append(float(synth_i_entry["TEMP"]))
            for a, axis in enumerate(axes.flat):
                if a == 0:
                    wvar = "PRECIP"
                    wvcolor = "royalblue"
                    axis.set_ylabel("Precipitation [m]")
                if a == 1:
                    wvar = "TEMP"
                    wvcolor = "firebrick"
                    axis.set_ylabel("Temperature ["+chr(176)+"C]")
                wvar = "PRECIP" if a == 0 else "TEMP"
                obs_p25, obs_p50, obs_p75 = [], [], []
                synth_p25, synth_p50, synth_p75 = [], [], []
                for d in obs_doy_dict.keys():
                    obs_p25.append(np.nanpercentile(obs_doy_dict[d][wvar], 25))
                    obs_p50.append(np.nanpercentile(obs_doy_dict[d][wvar], 50))
                    obs_p75.append(np.nanpercentile(obs_doy_dict[d][wvar], 75))
                for d in synth_doy_dict.keys():
                    synth_p25.append(np.nanpercentile(synth_doy_dict[d][wvar], 25))
                    synth_p50.append(np.nanpercentile(synth_doy_dict[d][wvar], 50))
                    synth_p75.append(np.nanpercentile(synth_doy_dict[d][wvar], 75))
                synth_doys, obs_doys = np.array(list(synth_doy_dict.keys())), np.array(list(obs_doy_dict.keys()))
                sorted_synth_doys = synth_doys[np.argsort(synth_doys)]
                sorted_synth_p25 = np.array(synth_p25)[np.argsort(synth_doys)]
                sorted_synth_p50 = np.array(synth_p50)[np.argsort(synth_doys)]
                sorted_synth_p75 = np.array(synth_p75)[np.argsort(synth_doys)]
                sorted_obs_doys = obs_doys[np.argsort(obs_doys)]
                sorted_obs_p25 = np.array(obs_p25)[np.argsort(obs_doys)]
                sorted_obs_p50 = np.array(obs_p50)[np.argsort(obs_doys)]
                sorted_obs_p75 = np.array(obs_p75)[np.argsort(obs_doys)]
                axis.fill_between(sorted_synth_doys, sorted_synth_p25, sorted_synth_p75, color="grey", alpha=0.33, zorder=10)
                axis.fill_between(sorted_obs_doys, sorted_obs_p25, sorted_obs_p75, color=wvcolor, alpha=0.33, zorder=11)
                axis.plot(sorted_synth_doys, sorted_synth_p50, color="grey", linestyle="-.", zorder=12)
                axis.plot(sorted_obs_doys, sorted_obs_p50, color="black", zorder=12)
            plt.tight_layout()
            doy_fig.savefig("{}Compare_PerDOY_{}.svg".format(dp, site.replace(" ", "")))
            plt.close()
