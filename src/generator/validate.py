import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


def validate_gmmhmm_states(min_states: int, max_states: int, lls: list[float], aics: list[float], bics: list[float]) -> None:
    """
    Validation figure for confirming the best-fitting number of states
    for the Gaussian mixture model hidden Markov model that represents
    precipitation

    Parameters
    ----------
    min_states: int
        The minimum number of attempted hidden states in fit
    max_states: int
        The maximum number of attempted hidden states in fit 
    lls: list[float]
        Log-likelihood calculation for each fit GMMHMM by number of states
    aics: list[float]
        AIC calculation for each fit GMMHMM by number of states
    bics: list[float]
        BIC calculation for each fit GMMHMM by number of states 
    """
    
    len_states = len(np.arange(min_states, max_states + 1))
    if len_states > len(lls):
        lls.extend([np.nan] * (len_states - len(lls)))
        aics.extend([np.nan] * (len_states - len(aics)))
        bics.extend([np.nan] * (len_states - len(bics)))
    num_states_plot, axis = plt.subplots()
    axis.grid() 
    axis.plot(np.arange(min_states, max_states + 1), aics, color="blue", marker="o", label="AIC")
    axis.plot(np.arange(min_states, max_states + 1), bics, color="green", marker="o", label="BIC")
    axis2 = axis.twinx()
    axis2.plot(np.arange(min_states, max_states + 1), lls, color="orange", marker="o", label="LL")
    axis.legend(handles=axis.lines + axis2.lines)
    axis.set_title("Validation of GMMHMM Best-Fitting Number of States")
    axis.set_xlabel("# States")
    axis.set_ylabel("Criterion Value [-, lower is better]")
    axis2.set_ylabel("Log-Likelihood [-, higher is better]")
    plt.tight_layout()
    num_states_plot.savefig("Validation_PrecipGMMHMM_NumStates.svg")
    plt.close()

