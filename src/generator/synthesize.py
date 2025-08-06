import numpy as np
import pandas as pd

from .make_figures import *


def synthesize_data(data: pd.DataFrame,
                    p_params: dict,
                    t_params: dict,
                    resolution: str,
                    validate: bool,
                    dirpath: str,
                    synthesize_kwargs: dict) -> pd.DataFrame:
    """
    Managing function that synthesizes weather from the fit or given 
    statistical parameters 

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe of formatted weather data to reference
    precip_params: dict
        Dictionary of precipitation parameters 
    copulaetemp_params: dict
        Dictionary of copula parameters to conditionally construct temperature 
    resolution: str
        The temporal resolution of the  data 
    validate: bool
        Flag for producing figures to validate each step of the generator
    dirpath: str
        Path for where to save the validation figures
    synthesize_kwargs: dict
        Keyword arguments related to the fit
    
    Returns
    -------
    synth_data: pd.DataFrame
        The synthesized weather data from the given observations and parameters
    """
    
    # validation
    global do_validation, validation_dirpath
    do_validation, validation_dirpath = validate, dirpath

    # synthesize kwargs    
    default_synthesize_kwargs = {}
    if not synthesize_kwargs: 
        synthesize_kwargs = default_synthesize_kwargs
    else:
        for k in default_synthesize_kwargs:
            if k not in synthesize_kwargs:
                synthesize_kwargs[k] = default_synthesize_kwargs[k]
    
    print(p_params)
    print(t_params)
    return pd.DataFrame()

