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
TRIAL="supervised_MR"
MODE="pred"   # or: avg
INPUT="odd"   # or: odd / even / both / all
RANGE="all"
# =======================================

# list of epochs you want to run
EPOCH_LIST=(185 115 85)

# loop through epochs
for EPOCH in "${EPOCH_LIST[@]}"; do
    echo "Running epoch $EPOCH ..."
    
    python3 predict_2D.py \
        --trial_name $TRIAL \
        --epoch $EPOCH \
        --mode $MODE \
        --input $INPUT \
        --slice_range $RANGE
done

echo "Finished all jobs."
