chimefrb-selection
==================

A Python package for modeling and applying the CHIME/FRB survey **selection function**.

This package provides:

- Construction of polynomial design matrices for logistic models
- Evaluation of fitted selection functions (``SelectionFunction``)
- A portable kNN envelope mask (``KNNEnvelope``) to restrict to in-support regions
- Pre-packaged model fits (``.npz``) and masks (``.pkl``) shipped in ``data/``

Installation
------------

Clone the repo and install in editable mode::

   git clone https://github.com/CHIMEFRB/chimefrb-selection.git
   cd chimefrb-selection
   pip install -e .

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   usage
   api

.. toctree::
   :maxdepth: 1
   :caption: Links

   GitHub repository <https://github.com/CHIMEFRB/chimefrb-selection>
