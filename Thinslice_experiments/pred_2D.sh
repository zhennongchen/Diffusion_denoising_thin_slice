#!/bin/bash

# # ============ USER SETTINGS ============
TRIAL="unsupervised_gaussian_brainCT_predict_noise"
MODE="pred"   # or: avg
OBJECTIVE="pred_noise"  # or: pred_x0
RANGE="30-80"
# =======================================

# list of epochs you want to run
EPOCH_LIST=(75) #150 97)
NFE_LIST=(10)

# loop through epochs then loop through NFE values
for EPOCH in "${EPOCH_LIST[@]}"; do
    echo "Running epoch $EPOCH ..."
    for NFE in "${NFE_LIST[@]}"; do
        echo "Running NFE $NFE ..."
    
    python3 predict_2D.py \
        --trial_name $TRIAL \
        --epoch $EPOCH \
        --mode $MODE \
        --objective $OBJECTIVE \
        --slice_range $RANGE \
        --NFE $NFE
done
done

# echo "Finished all jobs."
