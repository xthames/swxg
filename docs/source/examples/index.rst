Tutorial and Examples
=====================

The following material provides a step-by-step instruction set to understand how ``swxg`` works using the built-in test dataset.

Importing ``swxg``
-------------------

Importing everything from the ``swxg`` package is easy:

.. code-block:: python

    import swxg

The ``swxg`` package only has two methods: 

 * ``swxg.SWXGModel``: a class that fits the dataset, and from that can validate the fit, can generate new data, and can compare the generated data to the input data.
 * ``swxg.test_wx``: the test dataset that we'll work with for the tutorial. If you are familiar with ``swxg`` you do not need to import this.

The ``test_wx`` DataFrame
-------------------------

The ``swxg.test_wx`` dataset is a Pandas dataframe that when using ``print(swxg.test_wx)`` should look like this:

=====  ====  ==========  ========  =========
 ..    SITE   DATETIME    PRECIP     TEMP
=====  ====  ==========  ========  =========
  0     A    1922-01-01  0.042540  -5.733282
  1     A    1922-02-01  0.023077  -2.706253
  2     A    1922-03-01  0.024833   1.908417
  3     A    1922-04-01  0.026300   6.053633
  4     A    1922-05-01  0.019248  12.359490
...    ...      ...         ...       ...
13843   L    2023-08-01  0.043041  15.225745 
13844   L    2023-09-01  0.035799  11.344369 
13845   L    2023-10-01  0.032206   5.430705 
13846   L    2023-11-01  0.029645  -1.630817 
13847   L    2023-12-01  0.031300  -6.706674
=====  ====  ==========  ========  =========

.. |deg| unicode:: U+00B0
 
Let's parse this dataframe:

 * There are four columns: ``SITE``, ``DATETIME``, ``PRECIP``, and ``TEMP``. The ``swxg.SWXGModel`` class expects at least four columns with these names specifically, otherwise it won't know how to format, process, fit, or generate data. **It also requires this order for the columns as well**. In version 0.2.0 the generator will only generate precipitation and temperature, but in the future it may be able to do more.  
 * The ``SITE`` column has type ``str`` and has a unique identifier for each unique site. Letters ``A`` through ``L`` are used here, but full strings can also be used.
 * The ``DATETIME`` column has type ``datetime``. This is the standard object output using ``datetime.datetime.strptime(date_string, input_format_code)`` from the ``datetime`` package. In version 0.2.2 you must format the date using YYYY-MM-DD (so, hyphens, not forward slashes). You can format any ``strptime`` output with ``strftime`` and a corresponding ``output_format_code``.
 * The ``PRECIP`` column has type ``float``, and is reported in units of [m]. It is recommended that units are metric!
 * The ``TEMP`` column has type ``float``, and is reported in units of [\ |deg|\ C]. It is recommend that units are metric!

And that's it -- just a location, timestamp and linked precipitation and temperature are the only datapoints you need to get started (even with your own data)! If one of the columns has the wrong name, type, or the column is in the wrong location, the Python editor will throw an error until it is corrected and acceptable for the generator.

Applying an Input DataFrame to ``SWXGModel``
--------------------------------------------

Taking an input dataframe and priming the generator with it is trivial:

.. code-block:: python

    model = swxg.SWXGModel(swxg.test_wx)

This creates an instance of the ``SWXGModel`` class with ``test_wx`` as the initial input. **You cannot instantiate the class without an input dataframe**.

The ``SWXGModel`` class has the following (if initially empty) attributes:

 * ``raw_data``: This is the dataframe you gave it as input, exactly as it was input. This is here as a sanity check that your input successfully made it into the model without artifacts.
 * ``data``: This is the input dataframe, reformatted to separate the ``DATETIME`` column into ``YEARS``, ``MONTHS``, and potentially ``DAYS``, depending on the input dataframe resolution. Starting in version 0.2.2 you can include subdaily data but it will be aggregated to daily since a subdaily generation scheme has yet to be implemented.
 * ``resolution``: This is the resolution of the input dataframe as determined by the model. It can be ``daily`` or ``monthly``.
 * ``precip_fit_dict``: This is the dictionary of statistics related to fitting precipitation that the generator will use. It is initially ``{}``.
 * ``copulaetemp_fit_dict``: This is the dictionary of statistics related to fitting copulae and temperature that the generator will use. It is initially ``{}``.
 * ``is_fit``: This is a flag for whether or not the input data has been fit yet. **Generation cannot happen without the data having been previously fit**. It is initially ``False``.

Displaying ``model.data`` using ``print(model.data)`` should look like this:

=====  ====  ====  =====  ========  =========
 ..    SITE  YEAR  MONTH   PRECIP     TEMP
=====  ====  ====  =====  ========  =========
  0     A    1922    1    0.042540  -5.733282
  1     A    1922    2    0.023077  -2.706253
  2     A    1922    3    0.024833   1.908417
  3     A    1922    4    0.026300   6.053633
  4     A    1922    5    0.019248  12.359490
...    ...   ...    ...     ...       ...
13843   L    2023    8    0.043041  15.225745 
13844   L    2023    9    0.035799  11.344369 
13845   L    2023   10    0.032206   5.430705 
13846   L    2023   11    0.029645  -1.630817 
13847   L    2023   12    0.031300  -6.706674
=====  ====  ====  =====  ========  =========

and ``model.resolution == 'monthly'``. The determination of the ``monthly`` or ``daily`` resolution comes from the set of day values in the ``DATETIME`` raw data column. If you have multiple days in that column, the generator will assume you are inputting daily data. Picking a single day for all data---it doesn't matter which---will assume monthly data.

.. note::

    It is permissible to overwrite the model attributes, if you are comfortable with doing so and understand how fitting and/or generation works. **It is recommended that you do not** and let the generator do this for you.

Fitting Data
------------

Fitting the reformatted input data is as easy as:

.. code-block:: python

   model.fit()

Using the :meth:`fit() <swxg.SWXGModel.fit>` method will first fit the preciptation data and then the copula/temperature data. It returns nothing and only updates the internal attributes. You can confirm that both precipitation and copulas/temperature have been fit by (1) checking that ``model.is_fit == True`` and (2) observing the output to screen. The output to screen is a clean version of ``model.precip_fit_dict`` and ``model.copulaetemp_fit_dict`` and should look similar the following:

.. code-block:: text

  ----------- Precipitation Fit -----------
  * Number of GMMHMM States: 1

  * GMMHMM Means/Stds per Site and State
    STATE SITE     MEANS      STDS
  0       0    A -0.388914  0.102367
  1       0    B -0.455363  0.108115
  2       0    C -0.565228  0.092826
  3       0    D -0.674662  0.122653
  4       0    E -0.410840  0.114239
  5       0    F -0.705281  0.144569
  6       0    G -0.466940  0.080229
  7       0    H -0.437433  0.087906
  8       0    I -0.547517  0.101222
  9       0    J -0.436658  0.094661
  10      0    K -0.541077  0.105190
  11      0    L -0.387186  0.075588

  * Transition Probability Matrix
               TO STATE 0
  FROM STATE 0        1.0 
  -----------------------------------------
  
  -------------- Copulas Fit --------------
  Copula Statistics for: JAN
  * Best-Fitting Copula Family: Independence
  * All Family Parameters and Fit Comparison
                Hyperparameter       AIC Cramér von Mises Kolmogorov-Smirnov
  Independence             NaN  0.000000          0.02539            0.03515
  Frank              -0.428539  1.783427         0.030091           0.040788
  Gaussian           -0.017058  1.862079         0.017746           0.036266
  ...
  -----------------------------------------

.. |eacute| unicode:: U+00E9

The critical fitness statistics for precipitation are how many states were chosen by the GMMHMM, the means and standard deviations of the GMMHMM per site and state, and the transition probability matrix. These are fairly easy to interpret, though note that the precipitation data behind the scenes has been log\ :sub:`10`\ -transformed and so the means are negative and standard deviations reflect the transformation. The critical fitness statistics for the copulas are which month is being fit and the best fitting copula family using three different metrics (AIC, Cram\ |eacute|\ r von Mises, and Kolmogorov-Smirnov). Smaller numbers for all three metrics indicate better fitness, and any AIC value within 2 of another should be considered an equivalent fitness. In this case the Independence copula is the smallest across all three metrics and therefore it is determined to be the best choice. Note that the Cram\ |eacute|\ r von Mises and Kolmogorov-Smirnov metrics are bootstrapped and so there may be small differences between the values listed here and those on your display.

Several arguments and keyword arguments are accepted by the :meth:`fit() <swxg.SWXGModel.fit>` method. These include, but are not limited to, turning off the output statistics display (``verbose=False``), creating validation figures that step through the fitting process (``validate=True``), hard-setting the number of GMMHMM states to use (``kwargs={"gmmhmm_states": 1}``), and restricting the copula families to try (``kwargs={"copula_families": ["Frank"]}``). Please review the method to learn the default behavior and how to change it.

Generating (Synthesizing) Data
------------------------------

Generating data from the fit is just as easy as fitting the data:

.. code-block:: python

    wx = model.synthesize()

Using the :meth:`synthesize() <swxg.SWXGModel.synthesize>` method returns a dataframe of precipitation and temperature generated from the fit statistics. This method also takes several additional arguments which should be reviewed but are outside the scope of this Tutorial.

``print(wx)`` will have the general form:

=====  ====  ====  =====  ===============  ===============
 ..    SITE  YEAR  MONTH      PRECIP            TEMP
=====  ====  ====  =====  ===============  ===============
  0     A      1     1      p\ :sub:`1`      T\ :sub:`1`
  1     A      1     2      p\ :sub:`2`      T\ :sub:`2`
  2     A      1     3      p\ :sub:`3`      T\ :sub:`3`
  3     A      1     4      p\ :sub:`4`      T\ :sub:`4`
  4     A      1     5      p\ :sub:`5`      T\ :sub:`5`
...    ...   ...    ...         ...              ...
14683   L     102    8    p\ :sub:`14684`  T\ :sub:`14684` 
14684   L     102    9    p\ :sub:`14685`  T\ :sub:`14685` 
14685   L     102   10    p\ :sub:`14686`  T\ :sub:`14686` 
14686   L     102   11    p\ :sub:`14687`  T\ :sub:`14687` 
14687   L     102   12    p\ :sub:`14688`  T\ :sub:`14688`
=====  ====  ====  =====  ===============  ===============

This has the same format as the reformatted input dataframe, with some key differences: 

 * The ``YEAR`` column has been replaced with a value for the order in the sequence it was generated. This is because the generated data do not technically align to any input year.
 * The size of the dataframe increased. This is because generated data does not contain NaNs or empty rows, where the input dataset might. The generator will default to generating the number of years given to it in the input set unless otherwise specified by the ``n`` argument.
 * You can synthesize weather at as-fine or coarser resolutions than your input dataset using the ``resolution`` argument, but not finer. Attempting finer resolutions will default to the resolution of the input dataset.
 * The ``PRECIP`` and ``TEMP`` columns will be unique for each random seed. By `fixing the RNG seed <https://numpy.org/doc/2.2/reference/random/generator.html#numpy.random.Generator>`__ before fitting the input data you can guarantee reproducibility.

Next Steps
----------

And that's all there is to it! You can try generating a new sample of data by simply envoking ``wx2 = model.synthesize()``, or try fitting a dataset of your own. We recommend looking at :ref:`How to Interpret the Validation Figures <how-to-validate>` and the :ref:`API <api>` next. 
