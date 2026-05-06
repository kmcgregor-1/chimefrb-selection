# The CHIME/FRB Selection Function - chimefrb-selection
<img src="https://github.com/user-attachments/assets/a86ed9d5-33d7-45cd-9a1a-7ac165a5c481" 
    alt="SelectionFunctionREADMEArt" width="400" align="right" />

A Python package for modeling and applying the CHIME/FRB survey **selection function**

This package provides:

- Construction of polynomial design matrices for logistic models
- Evaluation of fitted selection functions (`SelectionFunction`)
- A portable kNN envelope mask (`KNNEnvelope`) to restrict to in-support regions
- Pre-packaged model fits (`.npz`) and masks (`.pkl`) shipped in `data/`

---


## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/CHIMEFRB/chimefrb-selection.git
cd chimefrb-selection
pip install -e .
