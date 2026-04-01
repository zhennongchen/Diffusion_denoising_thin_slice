#!/bin/bash

# # ============ USER SETTINGS ============
TRIAL="unsupervised_gaussian_brainCT"
MODE="pred"   # or: avg
RANGE="30-80"
# =======================================

# list of epochs you want to run
EPOCH_LIST=(61) #150 97)
NFE_LIST=(2 3 5 10 50)

# loop through epochs then loop through NFE values
for EPOCH in "${EPOCH_LIST[@]}"; do
    echo "Running epoch $EPOCH ..."
    for NFE in "${NFE_LIST[@]}"; do
        echo "Running NFE $NFE ..."
    
    python3 predict_2D.py \
        --trial_name $TRIAL \
        --epoch $EPOCH \
        --mode $MODE \
        --slice_range $RANGE \
        --NFE $NFE
done
done

echo "Finished all jobs."
