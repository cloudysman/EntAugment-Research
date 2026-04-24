#!/bin/bash

mkdir -p logs

RESULT_FILE="benchmark_composition_results.csv"
echo "=== Completing benchmark: 20 runs ==="
echo "=== Result file: $RESULT_FILE ==="

run2() {
    CMD1=$1; LOG1=$2
    CMD2=$3; LOG2=$4
    echo "[$(date '+%H:%M:%S')] START: $(basename $LOG1 .log) | $(basename $LOG2 .log)"
    nohup $CMD1 > $LOG1 2>&1 &
    PID1=$!
    nohup $CMD2 > $LOG2 2>&1 &
    PID2=$!
    wait $PID1 $PID2
    echo "[$(date '+%H:%M:%S')] DONE: $(grep 'BEST ACC' $LOG1 | tail -1)"
    echo "[$(date '+%H:%M:%S')] DONE: $(grep 'BEST ACC' $LOG2 | tail -1)"
}

run1() {
    CMD=$1; LOG=$2
    echo "[$(date '+%H:%M:%S')] START: $(basename $LOG .log)"
    nohup $CMD > $LOG 2>&1
    echo "[$(date '+%H:%M:%S')] DONE: $(grep 'BEST ACC' $LOG | tail -1)"
}

BASE="--dataset CIFAR100 --gpus 0 --result_file $RESULT_FILE"

# ════════════════════════════════════════════
# PHẦN 1: 2 runs thiếu — R50 seed 456
# ════════════════════════════════════════════
echo ""
echo "── [1/3] 2 runs thiếu: R50 seed 456 ──"

run2 \
    "python train_EntAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 456" \
    "logs/cutmix_c100_r50_s456.log" \
    "python train_EntAugment_EntCutMix.py $BASE --conf confs/resnet50.yaml --seed 456" \
    "logs/entcutmix_c100_r50_s456.log"

# ════════════════════════════════════════════
# PHẦN 2: 9 runs Pure EntAugment (critical baseline)
# ════════════════════════════════════════════
echo ""
echo "── [2/3] 9 runs Pure EntAugment (no CutMix) ──"

# R18
run2 \
    "python train_EntAugment.py $BASE --conf confs/resnet18.yaml --seed 42" \
    "logs/entaug_c100_r18_s42.log" \
    "python train_EntAugment.py $BASE --conf confs/resnet18.yaml --seed 123" \
    "logs/entaug_c100_r18_s123.log"

run1 \
    "python train_EntAugment.py $BASE --conf confs/resnet18.yaml --seed 456" \
    "logs/entaug_c100_r18_s456.log"

# R50
run2 \
    "python train_EntAugment.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/entaug_c100_r50_s42.log" \
    "python train_EntAugment.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/entaug_c100_r50_s123.log"

run1 \
    "python train_EntAugment.py $BASE --conf confs/resnet50.yaml --seed 456" \
    "logs/entaug_c100_r50_s456.log"

# WRN
run2 \
    "python train_EntAugment.py $BASE --conf confs/wrn2810.yaml --seed 42" \
    "logs/entaug_c100_wrn_s42.log" \
    "python train_EntAugment.py $BASE --conf confs/wrn2810.yaml --seed 123" \
    "logs/entaug_c100_wrn_s123.log"

run1 \
    "python train_EntAugment.py $BASE --conf confs/wrn2810.yaml --seed 456" \
    "logs/entaug_c100_wrn_s456.log"

# ════════════════════════════════════════════
# PHẦN 3: 9 runs Pure RandAugment (optional baseline)
# ════════════════════════════════════════════
echo ""
echo "── [3/3] 9 runs Pure RandAugment (no CutMix) ──"

# R18
run2 \
    "python train_RandAugment_only.py $BASE --conf confs/resnet18.yaml --seed 42" \
    "logs/randaug_c100_r18_s42.log" \
    "python train_RandAugment_only.py $BASE --conf confs/resnet18.yaml --seed 123" \
    "logs/randaug_c100_r18_s123.log"

run1 \
    "python train_RandAugment_only.py $BASE --conf confs/resnet18.yaml --seed 456" \
    "logs/randaug_c100_r18_s456.log"

# R50
run2 \
    "python train_RandAugment_only.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/randaug_c100_r50_s42.log" \
    "python train_RandAugment_only.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/randaug_c100_r50_s123.log"

run1 \
    "python train_RandAugment_only.py $BASE --conf confs/resnet50.yaml --seed 456" \
    "logs/randaug_c100_r50_s456.log"

# WRN
run2 \
    "python train_RandAugment_only.py $BASE --conf confs/wrn2810.yaml --seed 42" \
    "logs/randaug_c100_wrn_s42.log" \
    "python train_RandAugment_only.py $BASE --conf confs/wrn2810.yaml --seed 123" \
    "logs/randaug_c100_wrn_s123.log"

run1 \
    "python train_RandAugment_only.py $BASE --conf confs/wrn2810.yaml --seed 456" \
    "logs/randaug_c100_wrn_s456.log"

# ════════════════════════════════════════════
# Tổng kết
# ════════════════════════════════════════════
echo ""
echo "=== ALL DONE ==="
echo ""
column -t -s',' $RESULT_FILE