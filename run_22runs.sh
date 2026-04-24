#!/bin/bash

mkdir -p logs

RESULT_FILE="benchmark_composition_results.csv"
echo "=== 22-run Benchmark ==="
echo "=== Result file: $RESULT_FILE ==="
echo ""

# Helper: chạy 2 jobs song song, log ra file riêng
run2() {
    CMD1=$1; LOG1=$2
    CMD2=$3; LOG2=$4
    echo "[$(date '+%H:%M:%S')] START: $(basename $LOG1 .log) | $(basename $LOG2 .log)"
    nohup $CMD1 > $LOG1 2>&1 &
    PID1=$!
    nohup $CMD2 > $LOG2 2>&1 &
    PID2=$!
    wait $PID1 $PID2
    echo "[$(date '+%H:%M:%S')] DONE:  $(grep 'BEST ACC' $LOG1 | tail -1)"
    echo "[$(date '+%H:%M:%S')] DONE:  $(grep 'BEST ACC' $LOG2 | tail -1)"
}

# Helper: chạy 1 job đơn
run1() {
    CMD=$1; LOG=$2
    echo "[$(date '+%H:%M:%S')] START: $(basename $LOG .log)"
    nohup $CMD > $LOG 2>&1
    echo "[$(date '+%H:%M:%S')] DONE:  $(grep 'BEST ACC' $LOG | tail -1)"
}

BASE="--dataset CIFAR100 --gpus 0 --result_file $RESULT_FILE"

# ════════════════════════════════════════════
# PHẦN 1: 4 runs thiếu (EntAugCutMix + EntCutMix, R50, seed 42+123)
# ════════════════════════════════════════════
echo "── [1/4] 4 runs thiếu: R50 seed 42+123 ──"

run2 \
    "python train_EntAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/cutmix_c100_r50_s42.log" \
    "python train_EntAugment_EntCutMix.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/entcutmix_c100_r50_s42.log"

run2 \
    "python train_EntAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/cutmix_c100_r50_s123.log" \
    "python train_EntAugment_EntCutMix.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/entcutmix_c100_r50_s123.log"

# ════════════════════════════════════════════
# PHẦN 2: 9 runs Pure CutMix (3 archs × 3 seeds)
# ════════════════════════════════════════════
echo ""
echo "── [2/4] 9 runs Pure CutMix ──"

# R18
run2 \
    "python train_PureCutMix.py $BASE --conf confs/resnet18.yaml --seed 42" \
    "logs/purecutmix_c100_r18_s42.log" \
    "python train_PureCutMix.py $BASE --conf confs/resnet18.yaml --seed 123" \
    "logs/purecutmix_c100_r18_s123.log"

run1 \
    "python train_PureCutMix.py $BASE --conf confs/resnet18.yaml --seed 456" \
    "logs/purecutmix_c100_r18_s456.log"

# R50
run2 \
    "python train_PureCutMix.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/purecutmix_c100_r50_s42.log" \
    "python train_PureCutMix.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/purecutmix_c100_r50_s123.log"

run1 \
    "python train_PureCutMix.py $BASE --conf confs/resnet50.yaml --seed 456" \
    "logs/purecutmix_c100_r50_s456.log"

# WRN
run2 \
    "python train_PureCutMix.py $BASE --conf confs/wrn2810.yaml --seed 42" \
    "logs/purecutmix_c100_wrn_s42.log" \
    "python train_PureCutMix.py $BASE --conf confs/wrn2810.yaml --seed 123" \
    "logs/purecutmix_c100_wrn_s123.log"

run1 \
    "python train_PureCutMix.py $BASE --conf confs/wrn2810.yaml --seed 456" \
    "logs/purecutmix_c100_wrn_s456.log"

# ════════════════════════════════════════════
# PHẦN 3: 9 runs RandAugment + CutMix (3 archs × 3 seeds)
# ════════════════════════════════════════════
echo ""
echo "── [3/4] 9 runs RandAugment + CutMix ──"

# R18
run2 \
    "python train_RandAugment_CutMix.py $BASE --conf confs/resnet18.yaml --seed 42" \
    "logs/racutmix_c100_r18_s42.log" \
    "python train_RandAugment_CutMix.py $BASE --conf confs/resnet18.yaml --seed 123" \
    "logs/racutmix_c100_r18_s123.log"

run1 \
    "python train_RandAugment_CutMix.py $BASE --conf confs/resnet18.yaml --seed 456" \
    "logs/racutmix_c100_r18_s456.log"

# R50
run2 \
    "python train_RandAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/racutmix_c100_r50_s42.log" \
    "python train_RandAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/racutmix_c100_r50_s123.log"

run1 \
    "python train_RandAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 456" \
    "logs/racutmix_c100_r50_s456.log"

# WRN
run2 \
    "python train_RandAugment_CutMix.py $BASE --conf confs/wrn2810.yaml --seed 42" \
    "logs/racutmix_c100_wrn_s42.log" \
    "python train_RandAugment_CutMix.py $BASE --conf confs/wrn2810.yaml --seed 123" \
    "logs/racutmix_c100_wrn_s123.log"

run1 \
    "python train_RandAugment_CutMix.py $BASE --conf confs/wrn2810.yaml --seed 456" \
    "logs/racutmix_c100_wrn_s456.log"

# ════════════════════════════════════════════
# Tổng kết
# ════════════════════════════════════════════
echo ""
echo "=== ALL 22 RUNS DONE ==="
echo "Results saved to: $RESULT_FILE"
echo ""
column -t -s',' $RESULT_FILE