#!/bin/bash
# EPOCH=405
# MODE="avg"   # or: avg
# RANGE="100-200"
# echo "running input both ..."
# python3 predict_2D.py --trial_name unsupervised_gaussian --epoch $EPOCH --mode $MODE --input both --slice_range $RANGE
# epoch "running input all ..."
# python3 predict_2D.py --trial_name unsupervised_gaussian --epoch $EPOCH --mode $MODE --input all --slice_range $RANGE
# echo "Finished all jobs"

# # ============ USER SETTINGS ============
TRIAL="unsupervised_gaussian_brainCT_predict_noise"
MODE="pred"   # or: avg
INPUT="both"   # or: odd / even / both / all
RANGE="150-200"
# =======================================

# list of epochs you want to run
EPOCH_LIST=(140) #150 97)
NFE_LIST=(50)

# loop through epochs then loop through NFE values
for EPOCH in "${EPOCH_LIST[@]}"; do
    echo "Running epoch $EPOCH ..."
    for NFE in "${NFE_LIST[@]}"; do
        echo "Running NFE $NFE ..."
    
    python3 predict_2D.py \
        --trial_name $TRIAL \
        --epoch $EPOCH \
        --mode $MODE \
        --input $INPUT \
        --slice_range $RANGE \
        --NFE $NFE
    done

done

echo "Finished all jobs."
