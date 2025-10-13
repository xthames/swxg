import numpy as np
import pandas as pd
import datetime as dt

import pytest
from swxg.model import SWXGModel, test_wx


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_generator() -> None: 
    # IN SITU TEST DATA, MONTHLY RESOLUTION
    df = test_wx
    
    # # SYNTHETIC TEST DATA, (SUB)DAILY RESOLUTION
    # wx_dict = {}
    # sites = ["A", "B", "C"]
    # timestamps = np.arange(dt.datetime(2000, 1, 1), dt.datetime(2025, 10, 13), dt.timedelta(hours=6)).astype(dt.datetime)
    # i = 0
    # for site in sites:
    #     for timestamp in timestamps:
    #         precip_rand, temp_rand = np.random.rand() / 100., np.random.rand()
    #         precip = 0. if precip_rand < 0.007 else precip_rand 
    #         temp = (-12.5*np.cos(2*np.pi*float(timestamp.strftime("%j"))/365.25) + 12.5) + 5*temp_rand 
    #         wx_dict[i] = [site, timestamp, float(precip), float(temp)]
    #         i += 1
    # df = pd.DataFrame().from_dict(wx_dict, orient="index", columns=["SITE", "DATETIME", "PRECIP", "TEMP"]) 
    # df.reset_index(drop=True, inplace=True)
    # df.astype({"SITE": str, "DATETIME": "datetime64[ns]", "PRECIP": float, "TEMP": float})

    model = SWXGModel(df) 
    model.fit(validate=True, kwargs={"copula_families": ["Frank"])
    synth_wx = model.synthesize(validate=True)
