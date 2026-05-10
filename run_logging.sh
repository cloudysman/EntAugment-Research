#!/bin/bash

mkdir -p logs logs_dump

RESULT_FILE="benchmark_composition_results.csv"
DATASET="CIFAR100"
CONF="confs/resnet18.yaml"
GPU="0"
SEED="42"

echo "=== 4 Extended Logging Runs for Paper Analysis ==="
echo "=== Dataset: $DATASET | Arch: ResNet-18 | Seed: $SEED ==="
echo "=== Dump dir: logs_dump/ ==="
echo ""

run_log() {
    METHOD=$1
    echo "[$(date '+%H:%M:%S')] START: $METHOD"
    nohup python train_with_logging.py \
        --method $METHOD \
        --dataset $DATASET \
        --conf $CONF \
        --seed $SEED \
        --gpus $GPU \
        --result_file $RESULT_FILE \
        --dump_dir logs_dump \
        --grad_log_interval 10 \
        > logs/logging_${METHOD}_${DATASET}_r18_s${SEED}.log 2>&1
    BEST=$(grep "BEST ACC" logs/logging_${METHOD}_${DATASET}_r18_s${SEED}.log | tail -1)
    echo "[$(date '+%H:%M:%S')] DONE: $METHOD | $BEST"
}

# Chạy tuần tự 4 methods
# (song song sẽ conflict gradient logging buffer)

run_log EntAugment_CutMix
run_log RandAugment_CutMix
run_log EntAugment_only
run_log RandAugment_only

echo ""
echo "=== ALL 4 RUNS DONE ==="
echo ""
echo "Dump files:"
ls -lh logs_dump/
echo ""
echo "Results:"
column -t -s',' $RESULT_FILE