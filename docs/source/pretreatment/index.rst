.. _pretreat:

Pretreating Your Own Dataset
============================

``swxg`` is meant to accept any set of observations of precipitation and temperature, not just the test sets included with the library. However, in order to use ``swxg`` for this purpose, it is extremely likely that your input dataset will need to be cleaned (or "pretreated") before it can be fit and generated. This section will help you do just that.

Guiding Principles of Pretreatment
----------------------------------

.. |deg| unicode:: U+00B0

There are many ways to pretreat your data, and no method is intrinsically better than another. That said, there are three key ideas that you should keep in mind when pretreating data for ``swxg``:

 * The final dataset is a singular dataframe with (at least) four columns titled ``SITE``, ``DATETIME``, ``PRECIP``, ``TEMP``, in that order. The datatypes for those columns should be ``str`` or ``object``, ``datetime64[ns]``, ``float``, and ``float``, respectively.
 * Precipitation and temperature columns should be in units of [m] and [\ |deg|\ C], respectively.
 * Missing or incomplete precipitation and temperature entries should either be infilled from existing data/secondary sites or outright removed. This includes dates where there are observations but not for every site. The fitting process can actually handle missing data and incomplete data without issue. However, the generation scheme uses a non-parametric (*k*-NN) disaggregation algorithm when resolving data at finer resolutions, and therefore leaving missing or incomplete data in the observations can populate the generated data incompletely. Trying to produce validation figures for missing or incomplete data is not possible and may cause an error. 

Basic Pretreatment Procedure
----------------------------

To begin, let's assume that we have three datasets containing three columns: 

 1. datetime information with the format ``YYYY-MM-DD`` in a column labeled ``DT``
 2. precipitation in [inches] in a column labeled ``PRCP``
 3. temperature in [\ |deg|\ F] in a column labeled ``TAVG``

The datasets are named ``siteA.csv``, ``siteB.csv``, and ``siteC.csv``. You can use the following code to pretreat your own data using the principles above by replacing the terms where appropriate:

.. code-block:: python

    import numpy as np
    import pandas as pd    
    import datetime as dt

    raw_df = pd.DataFrame()
    for site in ["A", "B", "C"]:
        # read in dataset
        site_df = pd.read_csv("site{}.csv".format(site))
        
        # format columns
        site_df["SITE"] = site
        site_df = site_df[["SITE", "DT", "PRCP", "TEMP"]]
        site_df.rename(columns={"DT": "DATETIME", "PRCP": "PRECIP", "TAVG": "TEMP"})
        site_df["PRECIP"] = site_df["PRECIP"].values * 0.0254
        site_df["TEMP"] = (site_df["TEMP"].values - 32.) * (5./9)

        # stitch to single dataset
        raw_df = site_df if raw_df.empty else pd.concat([raw_df, site_df])
    
    # set datatypes
    raw_df.astype({"SITE": str, "DATETIME": str, "PRECIP": float, "TEMP": float})
    raw_df["DATETIME"] = pd.to_datetime(raw_df["DATETIME"])

If a small (read: please use your own best judgment here on what constitutes "small") amount of data is missing, you can infill from the existing dataset using averages of existing months or days. The code to do this looks like the following:

.. code-block:: python

    avg_dict = {}
    sites = sorted(set(raw_df["SITE"].values))

    # --- if input dataset is at MONTHLY resolution ---
    months = [int(pd.to_datetime(raw_df.iloc[i]["DATETIME"]).month) for i in range(raw_df.shape[0])]
    for site in sites:
        if site not in avg_dict: avg_dict[site] = {}
        site_idx = raw_df["SITE"] == site 
        site_entry = raw_df.loc[site_idx]
        for month in months:
            month_idx = [int(pd.to_datetime(raw_df.iloc[i]["DATETIME"]).month) == month for i in range(site_entry.shape[0])]
            avg_dict[site][month] = {"precip": [raw_df.loc[site_idx & month_idx, "PRECIP"].values],
                                     "temp": [raw_df.loc[site_idx & month_idx, "TEMP"].values]}
    for i in range(raw_df.shape[0]):
        row_entry = raw_df.iloc[i]
        site, month = row_entry["SITE"], int(pd.to_datetime(row_entry["DATETIME"]).month)
        if np.isnan(row_entry["PRECIP"]): raw_df.at[i, "PRECIP"] = float(np.nanmean(avg_dict[site][month]["precip"]))
        if np.isnan(row_entry["TEMP"]): raw_df.at[i, "TEMP"] = float(np.nanmean(avg_dict[site][month]["temp"]))

    # --- if input dataset is at DAILY resolution ---
    doys = [int(pd.to_datetime(raw_df.iloc[i]["DATETIME"]).dayofyear) for i in range(raw_df.shape[0])] 
    for site in sites:
        if site not in avg_dict: avg_dict[site] = {}
        site_idx = raw_df["SITE"] == site
        site_entry = raw_df.loc[site_idx]
        for i in range(site_entry.shape[0]):
            row_entry = site_entry.iloc[i]
            doy = int(pd.to_datetime(row_entry["DATETIME"]).dayofyear)
            if doy not in avg_dict[site]:
                avg_dict[site][doy] = {"precip": [row_entry["PRECIP"]], "temp": [row_entry["TEMP"]]}
            else:
                avg_dict[site][doy]["precip"].append(row_entry["PRECIP"])
                avg_dict[site][doy]["temp"].append(row_entry["TEMP"])
    for site in avg_dict:
        for doy in avg_dict[site]:
            avg_dict[site][doy]["precip"] = np.nanmean(avg_dict[site][doy]["precip"])
            avg_dict[site][doy]["temp"] = np.nanmean(avg_dict[site][doy]["temp"])
    for i in range(raw_df.shape[0]):
        row_entry = raw_df.iloc[i]
        site, doy = row_entry["SITE"], int(pd.to_datetime(row_entry["DATETIME"]).dayofyear)
        if np.isnan(row_entry["PRECIP"]): raw_df.at[i, "PRECIP"] = float(avg_dict[site][doy]["precip"])
        if np.isnan(row_entry["TEMP"]): raw_df.at[i, "TEMP"] = float(avg_dict[site][doy]["temp"]
    
    # remove periods when only some sites have data
    indices_to_remove = []
    for date in sorted(set(raw_df["DATETIME"].values)):
        date_idx = raw_df["DATETIME"] == date
        date_entry = raw_df.loc[date_idx]
        if date_entry.shape[0] != len(set(raw_df["SITE"].values)):
            indices_to_remove.append(int(date_entry.index[0]))
    clean_df = raw_df.drop(index=indices_to_remove)
    clean_df.reset_index(drop=True, inplace=True)
    
If too much of the dataset is missing or you cannot infill data from the existing/external sources, you can simply remove the offending entries. **Please be careful when bulk removing data as this may dramatically reduce the fitness of the model; referring to the validation figures is imperative when removing data**. The code to do this looks like the following:

.. code-block:: python

    # remove missing data
    dropped_missing_df = raw_df.dropna(axis=0)
    dropped_missing_df.reset_index(drop=True, inplace=True)

    # remove periods when only some sites have data
    indices_to_remove = []
    for date in sorted(set(dropped_missing_df["DATETIME"].values)):
        date_idx = dropped_missing_df["DATETIME"] == date
        date_entry = dropped_missing_df.loc[date_idx]
        if date_entry.shape[0] != len(set(dropped_missing_df["SITE"].values)):
            indices_to_remove.append(int(date_entry.index[0]))
    clean_df = dropped_missing_df.drop(index=indices_to_remove)
    clean_df.reset_index(drop=True, inplace=True)

Saving the cleaned dataframe is simple:
    
.. code-block:: python
    
    # save the dataframe -- .pkl is recommended because it saves datatypes and is always available in Python environments
    clean_df.to_pickle("clean_wx.pkl")


Alternative Procedures
----------------------

Additional data sources can occasionally be used to infill missing data. If using secondary sites to infill a primary, bias-correction of the secondary site(s) to the primary site(s) of interest is preferable to a deficient dataset. Bias-correction of hydroclimatic variables is a robust field, and you can find more information on how to do this `for precipitation <doi.org/10.1002/joc.2168>`__ and `for temperature <doi.org/10.1016/j.heliyon.2024.e40352>`__ at the linked sources. 
