#!/bin/bash
# Run all 1D predictor fits with fiducial marginalization
# This fits selection functions marginalized over the 3 unobserved parameters

set -e

FIDUCIAL_PARAMS="/data/user-data/kmcgregor/catalog2-outputs/param_values.dat"
ORDERS="1 2 3 4 5 6"
PREDICTORS=("fluence" "scattering_time" "width" "dm")

echo "=============================================="
echo "Running all 1D selection function fits"
echo "Marginalizing over 3 unobserved parameters"
echo "=============================================="

for PRED in "${PREDICTORS[@]}"; do
    for ORDER in $ORDERS; do
        echo ""
        echo "======================================================"
        echo "=== ${PRED}, order ${ORDER} ==="
        echo "======================================================"
        
        python logistic_regression_cli.py \
            --predictor "$PRED" \
            --order "$ORDER" \
            --snr-cut 12 \
            --sidelobe-cut 5.0 \
            --marginalize \
            --fiducial-params "$FIDUCIAL_PARAMS" \
            --weight-clip-min 1e-3 \
            --weight-clip-max 1e3 \
            --cross-validate \
            --cv-folds 5 \
            --plot
    done
done

echo ""
echo "=============================================="
echo "All 1D fits complete!"
echo "=============================================="