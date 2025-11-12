import numpy as np
import pandas as pd
import datetime as dt

import pytest
from swxg.model import SWXGModel
from swxg.test_wx import monthly, daily


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_generator() -> None:  
    #model = SWXGModel(monthly) 
    model = SWXGModel(daily) 
    model.fit()
    #synth_wx = model.synthesize()
    #print(synth_wx)
