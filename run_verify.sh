#!/bin/bash

LOG_DIR="logs_verify"
mkdir -p $LOG_DIR

echo "=== Verify Fair Comparison ==="
echo "Baseline (crop+flip only) vs Pure CutMix vs paper numbers"

# Chạy song song trên 1 GPU
nohup python train_Baseline.py \
    --dataset CIFAR100 --conf confs/resnet18.yaml \
    --seed 42 --gpus 0 \
    > $LOG_DIR/baseline_c100_r18.txt 2>&1 &
PID1=$!

nohup python train_PureCutMix.py \
    --dataset CIFAR100 --conf confs/resnet18.yaml \
    --seed 42 --gpus 0 \
    > $LOG_DIR/purecutmix_c100_r18.txt 2>&1 &
PID2=$!

echo "PIDs: Baseline=$PID1, PureCutMix=$PID2"
echo "Monitor: tail -f $LOG_DIR/baseline_c100_r18.txt"
echo "Monitor: tail -f $LOG_DIR/purecutmix_c100_r18.txt"

wait $PID1 $PID2

echo ""
echo "=== DONE ==="
echo "--- Baseline ---"
grep "BEST ACC" $LOG_DIR/baseline_c100_r18.txt | tail -1
echo "--- Pure CutMix ---"
grep "BEST ACC" $LOG_DIR/purecutmix_c100_r18.txt | tail -1
echo ""
echo "--- ablation_results.csv ---"
cat ablation_results.csv