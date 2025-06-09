from typing import List, Tuple
import numpy as np
import pandas as pd
import datetime as dt

from src.generator.model import SWXGModel


def test_base() -> None:
    n: int = 3650
    sites: List[str] = ["a"] * n
    dtstamps: List[dt.datetime] = [(dt.datetime.today() - dt.timedelta(days=n)) + dt.timedelta(days=i) for i in range(n)]
    precips: List[float] = np.random.rand(n)
    temps: List[float] = 5.*(np.random.rand(n)-0.5) + 10.
    
    df: pd.DataFrame = pd.DataFrame({"SITE": sites, 
                                     "DATETIME": dtstamps,
                                     "PRECIP": precips,
                                     "TEMP": temps})   

    model = SWXGModel(df)
    assert model.raw_data.equals(df)
    print(model.data)
    print(model.precip_fit_dict)
