from src.generator.model import SWXGModel

import numpy as np
import pandas as pd
import datetime as dt


def test_base() -> None:
    n: int = 100
    sites: list[str] = ["a"] * n
    dtstamps: list[dt.datetime] = [dt.datetime.today() + dt.timedelta(hours=i) for i in range(n)]
    precips: list[float] = np.random.rand(n)
    temps: list[float] = 5.*(np.random.rand(n)-0.5) + 10.
    
    df: pd.DataFrame = pd.DataFrame({"SITE": sites, 
                                     "DATETIME": dtstamps,
                                     "PRECIP": precips,
                                     "TEMP": temps})   

    model = SWXGModel(df)
    assert model.raw_data.equals(df)
    assert model.fit() == 1.
