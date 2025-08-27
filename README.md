# `swxg`
`swxg` is a Python package for modern [stochastic weather generation](https://www.ipcc-data.org/guidelines/pages/weather_generators.html). It quickly generates arbitrarily-long sequences of monthly or daily weather variables that match spatial and temporal correlations of input observations by: 
  1. fitting observed precipitation individually to a (Gaussian mixture model) hidden Markov model with 1 or more hidden states; 
  2. fitting both observed precipitation and temperature with hydroclimatic copulae;
  3. sampling precipitation from its fit, disaggregating to finer resolution where necessary, and;
  4. conditionally sampling temperature from the sampled precipitation and its fit, disaggregating to finer resolution where necessary

## Dependencies
The required dependencies to use `swxg` are:
  * `Python >= 3.10`
  * `copulae >= 0.8`
  * `copulas >= 0.12`
  * `hmmlearn >= 0.3`
  * `matplotlib >= 3.10`
  * `numpy >= 2.2`
  * `pandas >= 2.3`
  * `scikit-learn >= 1.7`
  * `scipy >= 1.15`
  * `statsmodels >= 0.14`

## Installation
To install `swxg` from PyPI with `pip`:

    pip install swxg

Alternatively, you can install from this repository:

    git clone https://github.com/xthames/swxg.git
    cd swxg
    pip install .

## Important Links
  * [Official Source Code](https://github.com/xthames/swxg)
  * [Documentation](https://swxg.readthedocs.org)

If your work uses `swxg`, please cite: 
  * [JOSS PAPER IN PREP], specifically for the software
  * [WRR PAPER IN PREP], if relevant to applied (first) use case

