#!/bin/bash

LOG_DIR="logs_scale_up"
mkdir -p $LOG_DIR

echo "=== Scale Up: H1 vs PureCutMix on R-50 and WRN-28-10 ==="
echo "=== 1 GPU, song song từng cặp ==="

# ── Cặp 1: R-50 ──
echo "[$(date '+%H:%M:%S')] START cặp R-50: H1 + PureCutMix song song"

nohup python train_EntCutMix_Ablation.py \
    --config H1 --dataset CIFAR100 \
    --conf confs/resnet50.yaml --seed 42 --gpus 0 \
    > $LOG_DIR/H1_c100_r50.txt 2>&1 &
PID1=$!

nohup python train_PureCutMix.py \
    --dataset CIFAR100 --conf confs/resnet50.yaml \
    --seed 42 --gpus 0 \
    > $LOG_DIR/purecutmix_c100_r50.txt 2>&1 &
PID2=$!

echo "  PIDs: H1=$PID1, PureCutMix=$PID2"
wait $PID1 $PID2

echo "[$(date '+%H:%M:%S')] DONE cặp R-50"
echo "  H1:         $(grep 'BEST ACC' $LOG_DIR/H1_c100_r50.txt | tail -1)"
echo "  PureCutMix: $(grep 'BEST ACC' $LOG_DIR/purecutmix_c100_r50.txt | tail -1)"

# ── Cặp 2: WRN-28-10 ──
echo "[$(date '+%H:%M:%S')] START cặp WRN-28-10: H1 + PureCutMix song song"

nohup python train_EntCutMix_Ablation.py \
    --config H1 --dataset CIFAR100 \
    --conf confs/wrn2810.yaml --seed 42 --gpus 0 \
    > $LOG_DIR/H1_c100_wrn.txt 2>&1 &
PID3=$!

nohup python train_PureCutMix.py \
    --dataset CIFAR100 --conf confs/wrn2810.yaml \
    --seed 42 --gpus 0 \
    > $LOG_DIR/purecutmix_c100_wrn.txt 2>&1 &
PID4=$!

echo "  PIDs: H1=$PID3, PureCutMix=$PID4"
wait $PID3 $PID4

echo "[$(date '+%H:%M:%S')] DONE cặp WRN-28-10"
echo "  H1:         $(grep 'BEST ACC' $LOG_DIR/H1_c100_wrn.txt | tail -1)"
echo "  PureCutMix: $(grep 'BEST ACC' $LOG_DIR/purecutmix_c100_wrn.txt | tail -1)"

# ── Tổng kết ──
echo ""
echo "=== ALL DONE ==="
echo ""
echo "Bảng kết quả:"
echo "Config     | R-18       | R-50       | WRN"
echo "-----------|------------|------------|------------"
echo "H1         | $(grep 'BEST ACC' logs_ablation/H1_CIFAR100_r18.txt 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)% | $(grep 'BEST ACC' $LOG_DIR/H1_c100_r50.txt | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)% | $(grep 'BEST ACC' $LOG_DIR/H1_c100_wrn.txt | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)%"
echo "PureCutMix | $(grep 'BEST ACC' logs_verify/purecutmix_c100_r18.txt 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)% | $(grep 'BEST ACC' $LOG_DIR/purecutmix_c100_r50.txt | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)% | $(grep 'BEST ACC' $LOG_DIR/purecutmix_c100_wrn.txt | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)%"
echo ""
cat ablation_results.csv