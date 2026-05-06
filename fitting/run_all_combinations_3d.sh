#!/bin/bash
# Run all 3D predictor combinations with fiducial marginalization
# This fits selection functions marginalized over the 1 unobserved parameter

set -e

FIDUCIAL_PARAMS="/data/user-data/kmcgregor/catalog2-outputs/param_values.dat"
ORDERS="1 2 3 4"
COMBINATIONS=(
    "fluence scattering_time width"       # marginalize over DM
    "fluence scattering_time dm"          # marginalize over width
    "fluence width dm"                    # marginalize over scattering_time
    "scattering_time width dm"            # marginalize over fluence
)

echo "=============================================="
echo "Running all 3D selection function fits"
echo "Marginalizing over unobserved parameter"
echo "=============================================="

for COMBO in "${COMBINATIONS[@]}"; do
    read -r PRED1 PRED2 PRED3 <<< "$COMBO"
    
    for ORDER in $ORDERS; do
        echo ""
        echo "======================================================"
        echo "=== ${PRED1} + ${PRED2} + ${PRED3}, order ${ORDER} ==="
        echo "======================================================"
        
        python logistic_regression_cli.py \
            --predictor "$PRED1" \
            --predictor "$PRED2" \
            --predictor "$PRED3" \
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
echo "All 3D fits complete!"
echo "=============================================="