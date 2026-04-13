#!/bin/bash
LOG_DIR="logs_benchmark_perBatch"
mkdir -p $LOG_DIR

run_one() {
    DATASET=$1; CONF=$2; SEED=$3; GPU=$4
    MODEL=$(basename $CONF .yaml)
    echo "[$(date +%H:%M:%S)] START: $DATASET $MODEL seed=$SEED"
    python train_EntAugment_MAB_perBatch.py \
        --dataset $DATASET --conf confs/$CONF \
        --seed $SEED --gpus $GPU \
        > $LOG_DIR/${DATASET}_${MODEL}_seed${SEED}.txt 2>&1
    BEST=$(grep "BEST ACC" $LOG_DIR/${DATASET}_${MODEL}_seed${SEED}.txt | tail -1)
    echo "[$(date +%H:%M:%S)] DONE: $DATASET $MODEL seed=$SEED | $BEST"
}

# CIFAR-10
for SEED in 42 123 456; do
    run_one CIFAR10 resnet18.yaml $SEED 0
done
for SEED in 42 123 456; do
    run_one CIFAR10 resnet50.yaml $SEED 0
done
for SEED in 42 123 456; do
    run_one CIFAR10 wrn2810.yaml $SEED 0
done

# CIFAR-100
for SEED in 42 123 456; do
    run_one CIFAR100 resnet18.yaml $SEED 0
done
for SEED in 42 123 456; do
    run_one CIFAR100 resnet50.yaml $SEED 0
done
for SEED in 42 123 456; do
    run_one CIFAR100 wrn2810.yaml $SEED 0
done

echo "=== ALL DONE ==="