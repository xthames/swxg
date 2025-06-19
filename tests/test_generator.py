from typing import List, Tuple
import numpy as np
import pandas as pd
import datetime as dt
import pytest
from src.generator.model import SWXGModel


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_generator() -> None:
    n: int = 3650
    sites: List[str] = ["a"] * n
    dtstamps: List[dt.datetime] = [(dt.datetime.today() - dt.timedelta(days=n)) + dt.timedelta(days=i) for i in range(n)]
    precips: List[float] = np.random.rand(n)/100.
    temps: List[float] = 10. + 10.*np.sin((2*np.pi/365.)*np.linspace(0, n, n)) + 5.*(np.random.rand(n)-0.5)
    
    df: pd.DataFrame = pd.DataFrame({"SITE": sites, 
                                     "DATETIME": dtstamps,
                                     "PRECIP": precips,
                                     "TEMP": temps})   

    model = SWXGModel(df)
    assert model.raw_data.equals(df)
    model.fit(fit_kwargs={"gmmhmm_max_states": 4})
    print(model.data)
    print(model.precip_fit_dict)
