import os

import numpy as np
import pandas as pd


class SWXGModel:
    """
    The base class to create, debias, fit, synthesize, and validate the stochastic 
    weather generation model.
    """

    def __init__(self, raw_data: pd.DataFrame) -> None:
        self.raw_data = raw_data


