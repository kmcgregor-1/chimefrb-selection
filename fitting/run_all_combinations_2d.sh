#!/bin/bash
# Run all 2D predictor combinations with fiducial marginalization
# This fits selection functions marginalized over the 2 unobserved parameters

set -e

FIDUCIAL_PARAMS="/data/user-data/kmcgregor/catalog2-outputs/param_values.dat"
ORDERS="1 2 3 4 5"
COMBINATIONS=(
    #"fluence scattering_time"
    #"fluence width"
    "fluence dm"
    #"scattering_time width"
    #"scattering_time dm"
    #"width dm"
)

echo "=============================================="
echo "Running all 2D selection function fits"
echo "Marginalizing over unobserved parameters"
echo "=============================================="

for COMBO in "${COMBINATIONS[@]}"; do
    read -r PRED1 PRED2 <<< "$COMBO"
    
    for ORDER in $ORDERS; do
        echo ""
        echo "======================================================"
        echo "=== ${PRED1} + ${PRED2}, order ${ORDER} ==="
        echo "======================================================"
        
        python logistic_regression_cli.py \
            --predictor "$PRED1" \
            --predictor "$PRED2" \
            --order "$ORDER" \
            --snr-cut 12 \
            --sidelobe-cut 5.0 \
            --marginalize \
            --fiducial-params "$FIDUCIAL_PARAMS" \
            --weight-clip-min 1e-3 \
            --weight-clip-max 1e3 \
            #--cross-validate \
            #--cv-folds 5 \
            #--plot
    done
done

echo ""
echo "=============================================="
echo "All 2D fits complete!"
echo "=============================================="