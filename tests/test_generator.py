import numpy as np
import pandas as pd
import datetime as dt

import pytest
from swxg.model import SWXGModel, test_wx


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_generator() -> None: 
    # TEST DATA, MONTHLY RESOLUTION
    df = test_wx
    model = SWXGModel(df)
    model.fit(validate=True, fit_kwargs={"copula_families": ["Frank"]})
    synth_wx = model.synthesize(resolution="monthly", validate=True)
