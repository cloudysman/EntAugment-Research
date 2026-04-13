#!/bin/bash
set -e
PYTHON=/home/hieudt/miniconda3/bin/python3.9
SCRIPT=train_Redundancy.py
GPU=0
LOG_DIR=logs/benchmark

mkdir -p $LOG_DIR

run() {
    DATASET=$1
    CONF=$2
    SEED=$3
    MODEL=$(basename $CONF .yaml)
    LOG="$LOG_DIR/${DATASET}_${MODEL}_seed${SEED}.txt"
    echo "[$(date '+%H:%M:%S')] START: $DATASET $MODEL seed=$SEED"
    $PYTHON $SCRIPT \
        --dataset $DATASET \
        --conf confs/$CONF \
        --gpus $GPU \
        --seed $SEED \
        > $LOG 2>&1
    BEST=$(grep "BEST ACC" $LOG | tail -1)
    echo "[$(date '+%H:%M:%S')] DONE:  $DATASET $MODEL seed=$SEED | $BEST"
}

# === CIFAR-10 ===
run CIFAR10  resnet18.yaml  42
run CIFAR10  resnet18.yaml  123
run CIFAR10  resnet18.yaml  456

run CIFAR10  resnet50.yaml  42
run CIFAR10  resnet50.yaml  123
run CIFAR10  resnet50.yaml  456

run CIFAR10  wrn2810.yaml   42
run CIFAR10  wrn2810.yaml   123
run CIFAR10  wrn2810.yaml   456

# === CIFAR-100 ===
run CIFAR100 resnet18.yaml  42
run CIFAR100 resnet18.yaml  123
run CIFAR100 resnet18.yaml  456

run CIFAR100 resnet50.yaml  42
run CIFAR100 resnet50.yaml  123
run CIFAR100 resnet50.yaml  456

run CIFAR100 wrn2810.yaml   42
run CIFAR100 wrn2810.yaml   123
run CIFAR100 wrn2810.yaml   456

echo "=== All 18 runs complete. Results saved to results.csv ==="
