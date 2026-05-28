#!/bin/bash

mkdir -p logs

RESULT_FILE="benchmark_results.csv"
DATASET="CIFAR10"
GPU="0"

echo "=== CIFAR-10 Full Benchmark: 54 runs ==="
echo "=== 6 methods × 3 archs × 3 seeds ==="
echo "=== Result file: $RESULT_FILE ==="
echo ""

run2() {
    CMD1=$1; LOG1=$2; CMD2=$3; LOG2=$4
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

BASE="--dataset $DATASET --gpus $GPU --result_file $RESULT_FILE"

# ════════════════════════════════════════════
# 1. EntAugment_CutMix — method chính
# ════════════════════════════════════════════
echo "── [1/6] EntAugment_CutMix ──"

run2 \
    "python train_EntAugment_CutMix.py $BASE --conf confs/resnet18.yaml --seed 42" \
    "logs/cutmix_c10_r18_s42.log" \
    "python train_EntAugment_CutMix.py $BASE --conf confs/resnet18.yaml --seed 123" \
    "logs/cutmix_c10_r18_s123.log"
run1 "python train_EntAugment_CutMix.py $BASE --conf confs/resnet18.yaml --seed 456" \
     "logs/cutmix_c10_r18_s456.log"

run2 \
    "python train_EntAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/cutmix_c10_r50_s42.log" \
    "python train_EntAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/cutmix_c10_r50_s123.log"
run1 "python train_EntAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 456" \
     "logs/cutmix_c10_r50_s456.log"

run2 \
    "python train_EntAugment_CutMix.py $BASE --conf confs/wrn2810.yaml --seed 42" \
    "logs/cutmix_c10_wrn_s42.log" \
    "python train_EntAugment_CutMix.py $BASE --conf confs/wrn2810.yaml --seed 123" \
    "logs/cutmix_c10_wrn_s123.log"
run1 "python train_EntAugment_CutMix.py $BASE --conf confs/wrn2810.yaml --seed 456" \
     "logs/cutmix_c10_wrn_s456.log"

# ════════════════════════════════════════════
# 2. PureCutMix — baseline chính
# ════════════════════════════════════════════
echo ""
echo "── [2/6] PureCutMix ──"

run2 \
    "python train_PureCutMix.py $BASE --conf confs/resnet18.yaml --seed 42" \
    "logs/purecutmix_c10_r18_s42.log" \
    "python train_PureCutMix.py $BASE --conf confs/resnet18.yaml --seed 123" \
    "logs/purecutmix_c10_r18_s123.log"
run1 "python train_PureCutMix.py $BASE --conf confs/resnet18.yaml --seed 456" \
     "logs/purecutmix_c10_r18_s456.log"

run2 \
    "python train_PureCutMix.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/purecutmix_c10_r50_s42.log" \
    "python train_PureCutMix.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/purecutmix_c10_r50_s123.log"
run1 "python train_PureCutMix.py $BASE --conf confs/resnet50.yaml --seed 456" \
     "logs/purecutmix_c10_r50_s456.log"

run2 \
    "python train_PureCutMix.py $BASE --conf confs/wrn2810.yaml --seed 42" \
    "logs/purecutmix_c10_wrn_s42.log" \
    "python train_PureCutMix.py $BASE --conf confs/wrn2810.yaml --seed 123" \
    "logs/purecutmix_c10_wrn_s123.log"
run1 "python train_PureCutMix.py $BASE --conf confs/wrn2810.yaml --seed 456" \
     "logs/purecutmix_c10_wrn_s456.log"

# ════════════════════════════════════════════
# 3. RandAugment_CutMix
# ════════════════════════════════════════════
echo ""
echo "── [3/6] RandAugment_CutMix ──"

run2 \
    "python train_RandAugment_CutMix.py $BASE --conf confs/resnet18.yaml --seed 42" \
    "logs/racutmix_c10_r18_s42.log" \
    "python train_RandAugment_CutMix.py $BASE --conf confs/resnet18.yaml --seed 123" \
    "logs/racutmix_c10_r18_s123.log"
run1 "python train_RandAugment_CutMix.py $BASE --conf confs/resnet18.yaml --seed 456" \
     "logs/racutmix_c10_r18_s456.log"

run2 \
    "python train_RandAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/racutmix_c10_r50_s42.log" \
    "python train_RandAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/racutmix_c10_r50_s123.log"
run1 "python train_RandAugment_CutMix.py $BASE --conf confs/resnet50.yaml --seed 456" \
     "logs/racutmix_c10_r50_s456.log"

run2 \
    "python train_RandAugment_CutMix.py $BASE --conf confs/wrn2810.yaml --seed 42" \
    "logs/racutmix_c10_wrn_s42.log" \
    "python train_RandAugment_CutMix.py $BASE --conf confs/wrn2810.yaml --seed 123" \
    "logs/racutmix_c10_wrn_s123.log"
run1 "python train_RandAugment_CutMix.py $BASE --conf confs/wrn2810.yaml --seed 456" \
     "logs/racutmix_c10_wrn_s456.log"

# ════════════════════════════════════════════
# 4. EntAugment_only
# ════════════════════════════════════════════
echo ""
echo "── [4/6] EntAugment_only ──"

run2 \
    "python train_EntAugment.py $BASE --conf confs/resnet18.yaml --seed 42" \
    "logs/entaug_c10_r18_s42.log" \
    "python train_EntAugment.py $BASE --conf confs/resnet18.yaml --seed 123" \
    "logs/entaug_c10_r18_s123.log"
run1 "python train_EntAugment.py $BASE --conf confs/resnet18.yaml --seed 456" \
     "logs/entaug_c10_r18_s456.log"

run2 \
    "python train_EntAugment.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/entaug_c10_r50_s42.log" \
    "python train_EntAugment.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/entaug_c10_r50_s123.log"
run1 "python train_EntAugment.py $BASE --conf confs/resnet50.yaml --seed 456" \
     "logs/entaug_c10_r50_s456.log"

run2 \
    "python train_EntAugment.py $BASE --conf confs/wrn2810.yaml --seed 42" \
    "logs/entaug_c10_wrn_s42.log" \
    "python train_EntAugment.py $BASE --conf confs/wrn2810.yaml --seed 123" \
    "logs/entaug_c10_wrn_s123.log"
run1 "python train_EntAugment.py $BASE --conf confs/wrn2810.yaml --seed 456" \
     "logs/entaug_c10_wrn_s456.log"

# ════════════════════════════════════════════
# 5. RandAugment_only
# ════════════════════════════════════════════
echo ""
echo "── [5/6] RandAugment_only ──"

run2 \
    "python train_RandAugment_only.py $BASE --conf confs/resnet18.yaml --seed 42" \
    "logs/randaug_c10_r18_s42.log" \
    "python train_RandAugment_only.py $BASE --conf confs/resnet18.yaml --seed 123" \
    "logs/randaug_c10_r18_s123.log"
run1 "python train_RandAugment_only.py $BASE --conf confs/resnet18.yaml --seed 456" \
     "logs/randaug_c10_r18_s456.log"

run2 \
    "python train_RandAugment_only.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/randaug_c10_r50_s42.log" \
    "python train_RandAugment_only.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/randaug_c10_r50_s123.log"
run1 "python train_RandAugment_only.py $BASE --conf confs/resnet50.yaml --seed 456" \
     "logs/randaug_c10_r50_s456.log"

run2 \
    "python train_RandAugment_only.py $BASE --conf confs/wrn2810.yaml --seed 42" \
    "logs/randaug_c10_wrn_s42.log" \
    "python train_RandAugment_only.py $BASE --conf confs/wrn2810.yaml --seed 123" \
    "logs/randaug_c10_wrn_s123.log"
run1 "python train_RandAugment_only.py $BASE --conf confs/wrn2810.yaml --seed 456" \
     "logs/randaug_c10_wrn_s456.log"

# ════════════════════════════════════════════
# 6. EntAugment_EntCutMix
# ════════════════════════════════════════════
echo ""
echo "── [6/6] EntAugment_EntCutMix ──"

run2 \
    "python train_EntAugment_EntCutMix.py $BASE --conf confs/resnet18.yaml --seed 42" \
    "logs/entcutmix_c10_r18_s42.log" \
    "python train_EntAugment_EntCutMix.py $BASE --conf confs/resnet18.yaml --seed 123" \
    "logs/entcutmix_c10_r18_s123.log"
run1 "python train_EntAugment_EntCutMix.py $BASE --conf confs/resnet18.yaml --seed 456" \
     "logs/entcutmix_c10_r18_s456.log"

run2 \
    "python train_EntAugment_EntCutMix.py $BASE --conf confs/resnet50.yaml --seed 42" \
    "logs/entcutmix_c10_r50_s42.log" \
    "python train_EntAugment_EntCutMix.py $BASE --conf confs/resnet50.yaml --seed 123" \
    "logs/entcutmix_c10_r50_s123.log"
run1 "python train_EntAugment_EntCutMix.py $BASE --conf confs/resnet50.yaml --seed 456" \
     "logs/entcutmix_c10_r50_s456.log"

run2 \
    "python train_EntAugment_EntCutMix.py $BASE --conf confs/wrn2810.yaml --seed 42" \
    "logs/entcutmix_c10_wrn_s42.log" \
    "python train_EntAugment_EntCutMix.py $BASE --conf confs/wrn2810.yaml --seed 123" \
    "logs/entcutmix_c10_wrn_s123.log"
run1 "python train_EntAugment_EntCutMix.py $BASE --conf confs/wrn2810.yaml --seed 456" \
     "logs/entcutmix_c10_wrn_s456.log"

# ════════════════════════════════════════════
# Tổng kết
# ════════════════════════════════════════════
echo ""
echo "=== ALL 54 RUNS DONE ==="
echo ""
python clean_and_summarize.py --dataset CIFAR10 2>/dev/null || \
    column -t -s',' $RESULT_FILE | grep CIFAR10