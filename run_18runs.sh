#!/bin/bash

mkdir -p logs

echo "=== 18 runs: EntAugment_CutMix + EntAugment_EntCutMix ==="
echo "=== CIFAR-100 x {R18, R50, WRN} x {seed 42, 123, 456} ==="
echo "=== Strategy: song song 2 jobs/lần, tuần tự theo architecture ==="

run2() {
    CMD1=$1; LOG1=$2; CMD2=$3; LOG2=$4
    echo "[$(date '+%H:%M:%S')] START: $(basename $LOG1) + $(basename $LOG2)"
    nohup $CMD1 > $LOG1 2>&1 &
    PID1=$!
    nohup $CMD2 > $LOG2 2>&1 &
    PID2=$!
    wait $PID1 $PID2
    echo "[$(date '+%H:%M:%S')] DONE:  $(grep 'BEST ACC' $LOG1 | tail -1)"
    echo "[$(date '+%H:%M:%S')] DONE:  $(grep 'BEST ACC' $LOG2 | tail -1)"
}

# ════════════════════════════════════════════
# ResNet-18 — nhẹ nhất, chạy trước
# ════════════════════════════════════════════
echo ""
echo "── ResNet-18 ──"

run2 \
    "python train_EntAugment_CutMix.py --dataset CIFAR100 --conf confs/resnet18.yaml --seed 42 --gpus 0" \
    "logs/cutmix_c100_r18_s42.log" \
    "python train_EntAugment_EntCutMix.py --dataset CIFAR100 --conf confs/resnet18.yaml --seed 42 --gpus 0" \
    "logs/entcutmix_c100_r18_s42.log"

run2 \
    "python train_EntAugment_CutMix.py --dataset CIFAR100 --conf confs/resnet18.yaml --seed 123 --gpus 0" \
    "logs/cutmix_c100_r18_s123.log" \
    "python train_EntAugment_EntCutMix.py --dataset CIFAR100 --conf confs/resnet18.yaml --seed 123 --gpus 0" \
    "logs/entcutmix_c100_r18_s123.log"

run2 \
    "python train_EntAugment_CutMix.py --dataset CIFAR100 --conf confs/resnet18.yaml --seed 456 --gpus 0" \
    "logs/cutmix_c100_r18_s456.log" \
    "python train_EntAugment_EntCutMix.py --dataset CIFAR100 --conf confs/resnet18.yaml --seed 456 --gpus 0" \
    "logs/entcutmix_c100_r18_s456.log"

# ════════════════════════════════════════════
# ResNet-50
# ════════════════════════════════════════════
echo ""
echo "── ResNet-50 ──"

run2 \
    "python train_EntAugment_CutMix.py --dataset CIFAR100 --conf confs/resnet50.yaml --seed 42 --gpus 0" \
    "logs/cutmix_c100_r50_s42.log" \
    "python train_EntAugment_EntCutMix.py --dataset CIFAR100 --conf confs/resnet50.yaml --seed 42 --gpus 0" \
    "logs/entcutmix_c100_r50_s42.log"

run2 \
    "python train_EntAugment_CutMix.py --dataset CIFAR100 --conf confs/resnet50.yaml --seed 123 --gpus 0" \
    "logs/cutmix_c100_r50_s123.log" \
    "python train_EntAugment_EntCutMix.py --dataset CIFAR100 --conf confs/resnet50.yaml --seed 123 --gpus 0" \
    "logs/entcutmix_c100_r50_s123.log"

run2 \
    "python train_EntAugment_CutMix.py --dataset CIFAR100 --conf confs/resnet50.yaml --seed 456 --gpus 0" \
    "logs/cutmix_c100_r50_s456.log" \
    "python train_EntAugment_EntCutMix.py --dataset CIFAR100 --conf confs/resnet50.yaml --seed 456 --gpus 0" \
    "logs/entcutmix_c100_r50_s456.log"

# ════════════════════════════════════════════
# WRN-28-10 — nặng nhất, chạy sau
# ════════════════════════════════════════════
echo ""
echo "── WRN-28-10 ──"

run2 \
    "python train_EntAugment_CutMix.py --dataset CIFAR100 --conf confs/wrn2810.yaml --seed 42 --gpus 0" \
    "logs/cutmix_c100_wrn_s42.log" \
    "python train_EntAugment_EntCutMix.py --dataset CIFAR100 --conf confs/wrn2810.yaml --seed 42 --gpus 0" \
    "logs/entcutmix_c100_wrn_s42.log"

run2 \
    "python train_EntAugment_CutMix.py --dataset CIFAR100 --conf confs/wrn2810.yaml --seed 123 --gpus 0" \
    "logs/cutmix_c100_wrn_s123.log" \
    "python train_EntAugment_EntCutMix.py --dataset CIFAR100 --conf confs/wrn2810.yaml --seed 123 --gpus 0" \
    "logs/entcutmix_c100_wrn_s123.log"

run2 \
    "python train_EntAugment_CutMix.py --dataset CIFAR100 --conf confs/wrn2810.yaml --seed 456 --gpus 0" \
    "logs/cutmix_c100_wrn_s456.log" \
    "python train_EntAugment_EntCutMix.py --dataset CIFAR100 --conf confs/wrn2810.yaml --seed 456 --gpus 0" \
    "logs/entcutmix_c100_wrn_s456.log"

# ════════════════════════════════════════════
# Tổng kết
# ════════════════════════════════════════════
echo ""
echo "=== ALL 18 RUNS DONE ==="
echo ""

summarize() {
    METHOD=$1; CONF=$2; TAG=$3
    S42=$(grep 'BEST ACC' logs/${METHOD}_c100_${TAG}_s42.log 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
    S123=$(grep 'BEST ACC' logs/${METHOD}_c100_${TAG}_s123.log 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
    S456=$(grep 'BEST ACC' logs/${METHOD}_c100_${TAG}_s456.log 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
    echo "  $METHOD/$CONF: seed42=$S42% seed123=$S123% seed456=$S456%"
}

echo "── EntAugment_CutMix ──"
summarize cutmix resnet18 r18
summarize cutmix resnet50 r50
summarize cutmix wrn2810  wrn

echo "── EntAugment_EntCutMix ──"
summarize entcutmix resnet18 r18
summarize entcutmix resnet50 r50
summarize entcutmix wrn2810  wrn