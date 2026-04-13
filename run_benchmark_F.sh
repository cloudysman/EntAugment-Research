#!/bin/bash

LOG_DIR="logs_benchmark_F"
mkdir -p $LOG_DIR

run_one() {
    DATASET=$1
    CONF=$2
    SEED=$3
    GPU=0
    MODEL=$(basename $CONF .yaml)

    echo "[$(date '+%H:%M:%S')] START: $DATASET $MODEL seed=$SEED"

    python train_EntAugment_MAB.py \
        --dataset $DATASET \
        --conf confs/$CONF \
        --seed $SEED \
        --gpus $GPU \
        > $LOG_DIR/${DATASET}_${MODEL}_seed${SEED}.txt 2>&1

    BEST=$(grep "BEST ACC" $LOG_DIR/${DATASET}_${MODEL}_seed${SEED}.txt | tail -1)
    echo "[$(date '+%H:%M:%S')] DONE:  $DATASET $MODEL seed=$SEED | $BEST"
}

echo "=== Benchmark Config F: EntAugment + MAB ==="
echo "=== Total: 18 runs, sequential on GPU 0 ==="

# === CIFAR-10 ===
run_one CIFAR10 resnet18.yaml 42
run_one CIFAR10 resnet18.yaml 123
run_one CIFAR10 resnet18.yaml 456

run_one CIFAR10 resnet50.yaml 42
run_one CIFAR10 resnet50.yaml 123
run_one CIFAR10 resnet50.yaml 456

run_one CIFAR10 wrn2810.yaml 42
run_one CIFAR10 wrn2810.yaml 123
run_one CIFAR10 wrn2810.yaml 456

# === CIFAR-100 ===
run_one CIFAR100 resnet18.yaml 42
run_one CIFAR100 resnet18.yaml 123
run_one CIFAR100 resnet18.yaml 456

run_one CIFAR100 resnet50.yaml 42
run_one CIFAR100 resnet50.yaml 123
run_one CIFAR100 resnet50.yaml 456

run_one CIFAR100 wrn2810.yaml 42
run_one CIFAR100 wrn2810.yaml 123
run_one CIFAR100 wrn2810.yaml 456

echo ""
echo "=== ALL DONE ==="
echo "Results saved in benchmark_results.csv"
cat benchmark_results.csv