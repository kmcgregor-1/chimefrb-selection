Usage
=====

This package provides tools for evaluating the CHIME/FRB survey selection
function.

Quickstart
----------

The main user-facing entry points are:

- :class:`chimefrb_selection.SelectionFunction` — load and evaluate a fitted
  selection function model
- :class:`chimefrb_selection.KNNEnvelope` — support mask (in log-space) used to
  restrict evaluation to in-support regions

SelectionFunction
-----------------

Create a default 4D selection function (fluence, scattering time, width, DM)::

  from chimefrb_selection import SelectionFunction

  sf = SelectionFunction()

Evaluate the selection probability at a single point (values are in linear
units; the model applies log10 transforms internally as needed)::

  props = {
      "fluence": 2.0,            # Jy ms
      "scattering_time": 0.01,   # seconds (or whatever units your fit expects)
      "width": 0.005,            # seconds
      "dm": 300.0,               # pc/cm^3
  }

  p = sf.calculate_selection_probability(props)
  print(p)

Vectorized evaluation
---------------------

You can pass arrays to evaluate many points at once::

  import numpy as np
  from chimefrb_selection import SelectionFunction

  sf = SelectionFunction()

  n = 100
  props = {
      "fluence": 10.0 ** np.random.uniform(-1, 2, n),
      "scattering_time": 10.0 ** np.random.uniform(-3, 0, n),
      "width": 10.0 ** np.random.uniform(-3, 0, n),
      "dm": 10.0 ** np.random.uniform(1, 3, n),
  }

  p = sf.calculate_selection_probability(props)

Predictor name aliases
----------------------

The package accepts some common aliases for predictor names (for convenience).
For example::

  props = {
      "fluence_jy_ms": 2.0,
      "tau_1_ghz_ms": 0.01,
      "pulse_width_ms": 5.0,
      "DM": 300.0,
  }

  p = sf.calculate_selection_probability(props)

See :data:`chimefrb_selection.utils.PREDICTOR_ALIASES` for the full mapping.

Notes on model files
--------------------

``SelectionFunction`` loads pre-fit model artifacts (``.npz``) and KNN masks
(``.pkl``) from the package data directory by default. These locations can be
overridden via environment variables:

- ``CHIMEFRB_SELECTION_MODELS_DIR``
- ``CHIMEFRB_SELECTION_KNN_DIR``
