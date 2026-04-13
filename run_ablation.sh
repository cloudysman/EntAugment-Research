#!/bin/bash

LOG_DIR="logs_ablation"
mkdir -p $LOG_DIR

DATASET="CIFAR100"
CONF="confs/resnet18.yaml"
GPU="0"

echo "=== EntCutMix Ablation: 5 configs, CIFAR100/ResNet-18 ==="

for CONFIG in A H1 H3 H4 H5; do
    echo "[$(date '+%H:%M:%S')] START: Config=$CONFIG"

    python train_EntCutMix_Ablation.py \
        --config $CONFIG \
        --dataset $DATASET \
        --conf $CONF \
        --seed 42 \
        --gpus $GPU \
        > $LOG_DIR/${CONFIG}_${DATASET}_r18.txt 2>&1

    BEST=$(grep "BEST ACC" $LOG_DIR/${CONFIG}_${DATASET}_r18.txt | tail -1)
    echo "[$(date '+%H:%M:%S')] DONE:  Config=$CONFIG | $BEST"
done

echo ""
echo "=== ALL DONE ==="
cat ablation_results.csv