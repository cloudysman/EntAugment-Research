#!/bin/bash
PYTHON=/home/hieudt/miniconda3/bin/python3.9
SCRIPT=train_Redundancy.py
GPU=0
LOG_DIR=logs/benchmark
mkdir -p $LOG_DIR

echo "[$(date '+%H:%M:%S')] === Starting benchmark (4 batches, 3 jobs/batch) ==="

# ── Batch 1: CIFAR10/wrn2810 x2 + CIFAR100/resnet18/42 ──
echo "[$(date '+%H:%M:%S')] Batch 1 START"
$PYTHON $SCRIPT --dataset CIFAR10  --conf confs/wrn2810.yaml   --gpus $GPU --seed 123 > $LOG_DIR/CIFAR10_wrn2810_seed123.txt   2>&1 & PID1=$!
$PYTHON $SCRIPT --dataset CIFAR10  --conf confs/wrn2810.yaml   --gpus $GPU --seed 456 > $LOG_DIR/CIFAR10_wrn2810_seed456.txt   2>&1 & PID2=$!
$PYTHON $SCRIPT --dataset CIFAR100 --conf confs/resnet18.yaml  --gpus $GPU --seed 42  > $LOG_DIR/CIFAR100_resnet18_seed42.txt  2>&1 & PID3=$!
echo "  PIDs: $PID1 $PID2 $PID3"
wait $PID1 $PID2 $PID3
echo "[$(date '+%H:%M:%S')] Batch 1 DONE"

# ── Batch 2: CIFAR100/resnet18 x2 + CIFAR100/resnet50/42 ──
echo "[$(date '+%H:%M:%S')] Batch 2 START"
$PYTHON $SCRIPT --dataset CIFAR100 --conf confs/resnet18.yaml  --gpus $GPU --seed 123 > $LOG_DIR/CIFAR100_resnet18_seed123.txt 2>&1 & PID4=$!
$PYTHON $SCRIPT --dataset CIFAR100 --conf confs/resnet18.yaml  --gpus $GPU --seed 456 > $LOG_DIR/CIFAR100_resnet18_seed456.txt 2>&1 & PID5=$!
$PYTHON $SCRIPT --dataset CIFAR100 --conf confs/resnet50.yaml  --gpus $GPU --seed 42  > $LOG_DIR/CIFAR100_resnet50_seed42.txt  2>&1 & PID6=$!
echo "  PIDs: $PID4 $PID5 $PID6"
wait $PID4 $PID5 $PID6
echo "[$(date '+%H:%M:%S')] Batch 2 DONE"

# ── Batch 3: CIFAR100/resnet50 x2 + CIFAR100/wrn2810/42 ──
echo "[$(date '+%H:%M:%S')] Batch 3 START"
$PYTHON $SCRIPT --dataset CIFAR100 --conf confs/resnet50.yaml  --gpus $GPU --seed 123 > $LOG_DIR/CIFAR100_resnet50_seed123.txt 2>&1 & PID7=$!
$PYTHON $SCRIPT --dataset CIFAR100 --conf confs/resnet50.yaml  --gpus $GPU --seed 456 > $LOG_DIR/CIFAR100_resnet50_seed456.txt 2>&1 & PID8=$!
$PYTHON $SCRIPT --dataset CIFAR100 --conf confs/wrn2810.yaml   --gpus $GPU --seed 42  > $LOG_DIR/CIFAR100_wrn2810_seed42.txt  2>&1 & PID9=$!
echo "  PIDs: $PID7 $PID8 $PID9"
wait $PID7 $PID8 $PID9
echo "[$(date '+%H:%M:%S')] Batch 3 DONE"

# ── Batch 4: CIFAR100/wrn2810 x2 ──
echo "[$(date '+%H:%M:%S')] Batch 4 START"
$PYTHON $SCRIPT --dataset CIFAR100 --conf confs/wrn2810.yaml   --gpus $GPU --seed 123 > $LOG_DIR/CIFAR100_wrn2810_seed123.txt 2>&1 & PID10=$!
$PYTHON $SCRIPT --dataset CIFAR100 --conf confs/wrn2810.yaml   --gpus $GPU --seed 456 > $LOG_DIR/CIFAR100_wrn2810_seed456.txt 2>&1 & PID11=$!
echo "  PIDs: $PID10 $PID11"
wait $PID10 $PID11
echo "[$(date '+%H:%M:%S')] Batch 4 DONE"

echo "[$(date '+%H:%M:%S')] === All 11 runs complete. Results in results.csv ==="
