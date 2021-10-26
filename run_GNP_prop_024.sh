#!/bin/bash
WEIGHT_DECAY=0.00001
DELTA=0.1

TRAIN_SET="${NUM_CONTEXT}-context_train"
TEST_SET="${NUM_CONTEXT}-context_test"

ROOT="_experiments/gnp_${NUM_CONTEXT}_context/${MODEL_NONDDP}/${MODEL_NONDDP}-prop-0.0"
echo $ROOT
mkdir -p $ROOT
(python -u train_pac.py --model $MODEL_NONDDP \
                    --num_threads 4 \
                    --dist_family full-cov \
                    --conv \
                    --points_per_unit 16 \
                    --internal_multiplier 1 \
                    --epochs 10 \
                    --learning_rate 0.001 \
                    --weight_decay $WEIGHT_DECAY \
                    --delta $DELTA \
                    --prior_proportion 0.0 \
                    --root $ROOT \
                    --train_set $TRAIN_SET \
                    --test_set $TEST_SET 2>&1 | tee "$ROOT/script_log.txt" )&

for PROP in 0.2 0.4
do
    ROOT="_experiments/gnp_${NUM_CONTEXT}_context/${MODEL_NONDDP}/${MODEL_NONDDP}-prop-$PROP"
    echo $ROOT
    mkdir -p $ROOT
    (python -u train_pac.py --model $MODEL_DDP \
                            --num_threads 4 \
                            --dist_family full-cov \
                            --conv \
                            --points_per_unit 16 \
                            --internal_multiplier 1 \
                            --epochs 10 \
                            --learning_rate 0.001 \
                            --weight_decay $WEIGHT_DECAY \
                            --delta $DELTA \
                            --prior_proportion $PROP \
                            --root $ROOT \
                            --train_set $TRAIN_SET \
                            --test_set $TEST_SET 2>&1 | tee "$ROOT/script_log.txt" )&
    sleep 1
done
