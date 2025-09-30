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

``test_wx``
-----------

The ``swxg.test_wx`` dataset is a Pandas dataframe that on display should look like this:

    =====  ====  ==========  ========  =========
      i    SITE   DATETIME    PRECIP      TEMP
    =====  ====  ==========  ========  =========
      0     A    1922-01-01  0.042540  -5.733282
      1     A    1922-02-01  0.023077  -2.706253
      2     A    1922-03-01  0.024833   1.908417
      3     A    1922-04-01  0.026300   6.053633
      4     A    1922-05-01  0.019248  12.359490
     ...   ...      ...         ...       ...
    13843   L    2023-08-01  0.043041  15.225745 
    13844   L    2023-09-01  0.035799  11.344369 
    13845   L    2023-10-01  0.032206   5.430705 
    13846   L    2023-11-01  0.029645  -1.630817 
    13847   L    2023-12-01  0.031300  -6.706674

.. |deg| unicode:: U+00B0
 
Let's parse this dataframe:

 * There are four columns: ``SITE``, ``DATETIME``, ``PRECIP``, and ``TEMP``. The ``swxg.SWXGModel`` class expects at least four columns with these names specifically, otherwise it won't know how to format, process, fit, or generate data. **It also requires this order for the columns as well**. In version 0.2.0 the generator will only generate precipitation and temperature, but in the future it may be able to do more.  
 * The ``SITE`` column has type ``str`` and has a unique identifier for each unique site. Letters ``A`` through ``L`` are used here, but full strings can also be used.
 * The ``DATETIME`` column has type ``datetime``. This is the standard object output using ``datetime.datetime.strptime(date_string, input_format_code)`` from the ``datetime`` package. In version 0.2.0 you must format the date using YYYY-MM-DD (so, hyphens, not forward slashes). You can format any ``strptime`` output with ``strftime`` and a corresponding ``output_format_code``.
 * The ``PRECIP`` column has type ``float``, and is reported in units of [m]. It is recommended that units are metric!
 * The ``TEMP`` column has type ``float``, and is reported in units of [\ |deg|\ C]. It is recommend that units are metric!
 
