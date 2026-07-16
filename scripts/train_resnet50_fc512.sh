#!/bin/bash
set -e
# Usage:
#   bash train_resnet50_fc512.sh [LR] [MAX_EPOCH] [DATA_FRAC]
#   bash train_resnet50_fc512.sh --resume <checkpoint_path> [extra_epochs]
# Defaults: LR=0.0003  MAX_EPOCH=3  DATA_FRAC=0.05  extra_epochs=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../coursework"

export SSL_CERT_FILE="C:/Users/soura/.conda/envs/reid/Lib/site-packages/certifi/cacert.pem"

ROOT="C:/Users/soura/code/2026/reid/data"

if [ "$1" = "--resume" ]; then
    CKPT="$2"
    EXTRA=${3:-10}
    [ -z "$CKPT" ] && { echo "Usage: $0 --resume <checkpoint_path> [extra_epochs]"; exit 1; }
    [ ! -f "$CKPT" ] && { echo "Checkpoint not found: $CKPT"; exit 1; }

    START=$(basename "$CKPT" | sed 's/model\.pth\.tar-//')
    MAX_EPOCH=$(( START + EXTRA ))
    CKPT_DIR=$(dirname "$CKPT")
    LR=$(basename "$CKPT_DIR" | grep -oE 'lr[0-9.e+-]+' | sed 's/lr//')
    DATA_FRAC=$(basename "$CKPT_DIR" | grep -oE 'frac[0-9.]+' | sed 's/frac//')
    LR=${LR:-0.0003}
    DATA_FRAC=${DATA_FRAC:-0.05}

    WARMUP=$(( MAX_EPOCH / 3 )); [ "$WARMUP" -lt 1 ] && WARMUP=1
    STEP1=$(( MAX_EPOCH * 2 / 3 )); [ "$STEP1" -lt 2 ] && STEP1=2
    STEP2=$(( MAX_EPOCH * 5 / 6 )); [ "$STEP2" -le "$STEP1" ] && STEP2=$(( STEP1 + 1 ))
    EVAL_FREQ=$(( EXTRA / 5 )); [ "$EVAL_FREQ" -lt 1 ] && EVAL_FREQ=1
    PATIENCE=$(( EXTRA / 4 )); [ "$PATIENCE" -lt 0 ] && PATIENCE=0

    echo "Resuming from $CKPT (epoch $START) -> $MAX_EPOCH (extra=$EXTRA)"
    echo "  LR=$LR  DATA_FRAC=$DATA_FRAC  WARMUP=$WARMUP  STEPS=$STEP1,$STEP2  EVAL_FREQ=$EVAL_FREQ"

    STUDENT_ID=ss50456 STUDENT_NAME="Sourav Sen" python main.py \
        -s veri -t veri \
        -a resnet50_fc512 \
        --root "$ROOT" \
        --height 224 --width 224 \
        --optim amsgrad --lr "$LR" \
        --max-epoch "$MAX_EPOCH" \
        --stepsize "$STEP1" "$STEP2" \
        --eval-freq "$EVAL_FREQ" \
        --patience "$PATIENCE" \
        --train-batch-size 64 \
        --test-batch-size 100 \
        --workers 0 \
        --data-fraction "$DATA_FRAC" \
        --train-sampler RandomIdentitySampler \
        --num-instances 4 \
        --lr-scheduler multi_step_warmup \
        --warmup-epochs "$WARMUP" \
        --save-checkpoint \
        --resume "$CKPT" \
        --save-dir "$CKPT_DIR"
    exit 0
fi

if [ $# -eq 0 ]; then
    read -p "LR [0.0003]: " _in; LR=${_in:-0.0003}
    read -p "MAX_EPOCH [3]: " _in; MAX_EPOCH=${_in:-3}
    read -p "DATA_FRAC [0.05]: " _in; DATA_FRAC=${_in:-0.05}
else
    LR=${1:-0.0003}
    MAX_EPOCH=${2:-3}
    DATA_FRAC=${3:-0.05}
fi

SAVE_DIR="logs/$(date +%d%m_%H%M)/train_resnet50_fc512_lr${LR}_ep${MAX_EPOCH}_frac${DATA_FRAC}"

WARMUP=$(( MAX_EPOCH / 3 )); [ "$WARMUP" -lt 1 ] && WARMUP=1
STEP1=$(( MAX_EPOCH * 2 / 3 )); [ "$STEP1" -lt 2 ] && STEP1=2
STEP2=$(( MAX_EPOCH * 5 / 6 )); [ "$STEP2" -le "$STEP1" ] && STEP2=$(( STEP1 + 1 ))
EVAL_FREQ=$(( MAX_EPOCH / 5 )); [ "$EVAL_FREQ" -lt 1 ] && EVAL_FREQ=1
PATIENCE=$(( MAX_EPOCH / 4 )); [ "$PATIENCE" -lt 0 ] && PATIENCE=0

STUDENT_ID=ss50456 STUDENT_NAME="Sourav Sen" python main.py \
    -s veri -t veri \
    -a resnet50_fc512 \
    --root "$ROOT" \
    --height 224 --width 224 \
    --optim amsgrad --lr "$LR" \
    --max-epoch "$MAX_EPOCH" \
    --stepsize "$STEP1" "$STEP2" \
    --eval-freq "$EVAL_FREQ" \
    --patience "$PATIENCE" \
    --train-batch-size 64 \
    --test-batch-size 100 \
    --workers 0 \
    --data-fraction "$DATA_FRAC" \
    --train-sampler RandomIdentitySampler \
    --num-instances 4 \
    --lr-scheduler multi_step_warmup \
    --warmup-epochs "$WARMUP" \
    --save-checkpoint \
    --save-dir "$SAVE_DIR"
