from typing import List, Tuple
import numpy as np
import pandas as pd
import datetime as dt
import calendar

import pytest
from src.generator.model import SWXGModel


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_generator() -> None:
    # REAL DATA
    df = pd.read_csv("/storage/home/ayt5134/work/research/SWG/swxg/tests/NOAA_UCRBMonthly.csv")
    df.drop(["LAT", "LON", "ELEV", "TMIN", "TMAX"], axis=1, inplace=True) 
    df.rename(columns={"NAME": "SITE", "PRCP": "PRECIP", "TAVG": "TEMP"}, inplace=True)
    datetime_col = []
    for i in range(df.shape[0]):
        i_entry = df.iloc[i]
        datetime_col.append(dt.datetime.strptime("{}-{}".format(i_entry["YEAR"], i_entry["MONTH"]), "%Y-%b"))
    df.insert(1, "DATETIME", datetime_col)
    df.drop(["YEAR", "MONTH"], axis=1, inplace=True)
    
    # # -- ARTIFICIAL DATA
    # n: int = int(365.25 * 50)
    # site_names = ["A", "B", "C"]
    # sites: List[str] = []
    # for site_set in [[site_name]*n for site_name in site_names]:
    #     sites.extend(site_set)
    # dtstamps: List[dt.datetime] = [(dt.datetime.today() - dt.timedelta(days=n)) + dt.timedelta(days=i) for i in range(n)]*len(site_names)
    # precips: List[float] = []
    # temps: List[float] = []
    # for i in range(len(site_names)):
    #     ps = (np.random.rand(n) - 0.5)/100.
    #     ps[ps < 0] = 0.
    #     precips.extend(ps)
    #     temps.extend(10. + 10.*np.sin((2*np.pi/365.)*np.linspace(0, n, n)) + 5.*(np.random.rand(n)-0.5))
    # 
    # df: pd.DataFrame = pd.DataFrame({"SITE": sites, 
    #                                  "DATETIME": dtstamps,
    #                                  "PRECIP": precips,
    #                                  "TEMP": temps})   

    model = SWXGModel(df)
    model.fit(validate=False, fit_kwargs={"copula_families": ["Frank"]})
    synth_wx = model.synthesize(resolution="daily", validate=True)

