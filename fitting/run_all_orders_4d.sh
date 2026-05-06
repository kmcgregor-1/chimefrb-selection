#!/usr/bin/env bash
# Run logistic_regression_cli.py for all polynomial orders with all four predictors

PYTHON=python
SCRIPT=logistic_regression_cli.py
INJ_FILE=/data/user-data/kmcgregor/09-2025_injections/output.json
BASE_PATH=../chimefrb_selection/data/fits
SNR_CUT=15
SIDELOBE_CUT=5
FIDUCIAL_MODEL_PATH=/data/user-data/kmcgregor/catalog2-outputs/param_values.dat

echo "=== Running order 1 ==="
$PYTHON $SCRIPT --predictor fluence --predictor scattering_time --predictor width --predictor dm \
  --order 1 --snr-cut $SNR_CUT --sidelobe-cut $SIDELOBE_CUT \
  --inj-file $INJ_FILE --base-path $BASE_PATH --marginalize --fiducial-params $FIDUCIAL_MODEL_PATH \
  --cross-validate --cv-folds 5

echo "=== Running order 2 ==="
$PYTHON $SCRIPT --predictor fluence --predictor scattering_time --predictor width --predictor dm \
  --order 2 --snr-cut $SNR_CUT --sidelobe-cut $SIDELOBE_CUT \
  --inj-file $INJ_FILE --base-path $BASE_PATH  --no-marginalize --fiducial-params $FIDUCIAL_MODEL_PATH \
  --cross-validate --cv-folds 5

echo "=== Running order 3 ==="
$PYTHON $SCRIPT --predictor fluence --predictor scattering_time --predictor width --predictor dm \
  --order 3 --snr-cut $SNR_CUT --sidelobe-cut $SIDELOBE_CUT \
  --inj-file $INJ_FILE --base-path $BASE_PATH --marginalize --fiducial-params $FIDUCIAL_MODEL_PATH \
 --cross-validate --cv-folds 5

echo "=== Running order 4 ==="
$PYTHON $SCRIPT --predictor fluence --predictor scattering_time --predictor width --predictor dm \
  --order 4 --snr-cut $SNR_CUT --sidelobe-cut $SIDELOBE_CUT \
  --inj-file $INJ_FILE --base-path $BASE_PATH --marginalize --fiducial-params $FIDUCIAL_MODEL_PATH \
  --cross-validate --cv-folds 5

echo "=== Running order 5 ==="
$PYTHON $SCRIPT --predictor fluence --predictor scattering_time --predictor width --predictor dm \
  --order 5 --snr-cut $SNR_CUT --sidelobe-cut $SIDELOBE_CUT \
  --inj-file $INJ_FILE --base-path $BASE_PATH --marginalize --fiducial-params $FIDUCIAL_MODEL_PATH \
  --cross-validate --cv-folds 5

echo "=== Running order 6 ==="
$PYTHON $SCRIPT --predictor fluence --predictor scattering_time --predictor width --predictor dm \
  --order 6 --snr-cut $SNR_CUT --sidelobe-cut $SIDELOBE_CUT \
  --inj-file $INJ_FILE --base-path $BASE_PATH --marginalize --fiducial-params $FIDUCIAL_MODEL_PATH \
  --cross-validate --cv-folds 5

echo "=== Running order 7 ==="
$PYTHON $SCRIPT --predictor fluence --predictor scattering_time --predictor width --predictor dm \
  --order 7 --snr-cut $SNR_CUT --sidelobe-cut $SIDELOBE_CUT \
  --inj-file $INJ_FILE --base-path $BASE_PATH --marginalize --fiducial-params $FIDUCIAL_MODEL_PATH \
  --cross-validate --cv-folds 5

echo "=== Running order 8 ==="
$PYTHON $SCRIPT --predictor fluence --predictor scattering_time --predictor width --predictor dm \
  --order 8 --snr-cut $SNR_CUT --sidelobe-cut $SIDELOBE_CUT \
  --inj-file $INJ_FILE --base-path $BASE_PATH --marginalize --fiducial-params $FIDUCIAL_MODEL_PATH \
  --cross-validate --cv-folds 5

echo "=== All runs complete! ==="
