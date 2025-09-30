How to Interpret the Validation Figures
=======================================

When you eventually want to look at the validation figures, the sheer number can be overwhelming. This page will help you make sense of each of these, and from which function they derive.

.. note::

   In both the :meth:`fit() <swxg.SWXGModel.fit>` and :meth:`synthesize() <swxg.SWXGModel.synthesize>` methods there is an argument called ``dirpath``. You can give it a string *relative to the current working directory* and that is where all of the figures created by that method will go. Setting the path in one method will not set it in the other; this is to give the user more control over where these figures end up.

.. note::

   All of the validation figures have ``.svg`` extensions. This is so that you can zoom in on the figures and pick things apart by eye, or remove elements for clarity in the appropriate editor.


:func:`validate_gmmhmm_states() <swxg.make_figures.validate_gmmhmm_states>`
---------------------------------------------------------------------------

 * **How to Interpret**: This function tests how good the GMMHMM fit is for each of the explored number of states. Note that the loglikelihood statistic is monotonically increasing and the AIC statistic can run into issues with overfitting, so the BIC is generally the best choice. If you set the number of states and don't explore any this will not be called.
 * **Nested Functions**: N/A
 * **Output**: ``Validate_GMMHMM_NumStates.svg``
