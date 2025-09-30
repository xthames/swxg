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

The ``swxg.test_wx`` dataset is a Pandas dataframe that on display should look like this:

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
 * The ``DATETIME`` column has type ``datetime``. This is the standard object output using ``datetime.datetime.strptime(date_string, input_format_code)`` from the ``datetime`` package. In version 0.2.0 you must format the date using YYYY-MM-DD (so, hyphens, not forward slashes). You can format any ``strptime`` output with ``strftime`` and a corresponding ``output_format_code``.
 * The ``PRECIP`` column has type ``float``, and is reported in units of [m]. It is recommended that units are metric!
 * The ``TEMP`` column has type ``float``, and is reported in units of [\ |deg|\ C]. It is recommend that units are metric!

If one of the columns has the wrong name, type, or the column is in the wrong location, the Python editor will throw an error until it is corrected and acceptable for the generator.

Applying an Input DataFrame to ``SWXGModel``
--------------------------------------------

Taking an input dataframe and priming the generator with it is trivial:

.. code-block:: python

    model = swxg.SWXGModel(swxg.test_wx)

This creates an instance of the ``SWXGModel`` class with ``test_wx`` as the initial input. **In order to instantiate the class you must include an input dataframe**.

The ``SWXGModel`` class has the following (if initially empty) attributes:

 * ``raw_data``: This is the dataframe you gave it as input, exactly as it was input. This is here as a sanity check that your input successfully made it into the model without artifacts.
 * ``data``: This is the input dataframe, reformatted to separate the ``DATETIME`` column into ``YEARS``, ``MONTHS``, and potentially ``DAYS``, depending on the input dataframe resolution.
 * ``resolution``: This is the resolution of the input dataframe as determined by the model. It can be ``daily`` or ``monthly``.
 * ``precip_fit_dict``: This is the dictionary of statistics related to fitting precipitation that the generator will use. It is initially ``{}``.
 * ``copulaetemp_fit_dict``: This is the dictionary of statistics related to fitting copulae and temperature that the generator will use. It is initially ``{}``.
 * ``is_fit``: This is a flag for whether or not the input data has been fit yet. **Generation cannot happen without the data having been previously fit**. It is initially ``False``.

Displaying ``model.data`` should look like this:

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

and ``model.resolution == 'monthly'``. Note that the determination of the ``monthly`` or ``daily`` resolution comes from the set of day values in the ``DATETIME`` raw data column. If you have multiple days in that column, the generator will assume you are inputting daily data. Picking a single day for all data---it doesn't matter which---will assume monthly data.

.. note::

    It is permissible to overwrite the model attributes, if you are comfortable with doing so and understand how fitting and/or generation works. **It is recommended that you do not** and let the generator do this for you.

Fitting Data
------------

Fitting the reformatted input data is as easy as:

.. code-block:: python

   model.fit(validate=False)

Using the ``fit()`` method will first fit the preciptation data and then the temperature data. It return nothing and only updates the internal attributes. You can confirm that both precipitation and temperature have been fit by checking that the ``precip_fit_dict`` and ``copulaetemp_fit_dict`` dictionaries have been populated, and that ``is_fit`` has been flipped to ``True``. :meth:`This method takes several arguments <swxg.SWXGModel.fit>` and one of them is ``validate``, which creates figures to visualize how the fitting is done. In this Tutorial we do not need to validate the test data so we set that ``validate=False``, but the default behavior is to produce these figures.

Generating (Synthesizing) Data
------------------------------

Generating data from the fit is just as easy as fitting the data:

.. code-block:: python

    wx = model.synthesize(validate=False)

Using the ``synthesize()`` method returns a dataframe of precipitation and temperature generated from the fit statistics. Again we set ``validate=False``, which defaults to creating figures to visualize how the generated data compares to the input data. This method also takes several additional arguments which are outside the scope of this Tutorial.

Displaying ``wx`` will look something like:

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
 * The size of the dataframe increased. This is because generated data does not contain NaNs or empty rows, where in input dataset might. The generator will default to generating the number of years given to it in the input set unless otherwise specified by the ``n`` argument.
 * The ``PRECIP`` and ``TEMP`` columns will be unique for each random seed. By `fixing the RNG seed <https://numpy.org/doc/2.2/reference/random/generator.html#numpy.random.Generator>`__ before fitting the input data you can guarantee reproducibility.
