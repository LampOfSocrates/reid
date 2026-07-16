#!/bin/bash
set -e
# Usage:
#   bash scripts/train.sh <arch>  [--lr LR] [--bs BS] [--max-epoch N] [--data-fraction F] [--label-smooth] [aug flags...]
#   bash scripts/train.sh <arch>  --resume <checkpoint> [--extra N]
#
# Augmentation flags are passed through to main.py unchanged:
#   --random-erase  --color-jitter  --blur-aug  --color-aug
#
# Scheduler (chosen automatically by arch):
#   swin_t_fc512, deit_s_fc512 -> cosine            (warmup = epoch/10)
#   everything else            -> multi_step_warmup  (warmup = epoch/3, steps at 2/3 and 5/6)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../coursework"

export SSL_CERT_FILE="C:/Users/soura/.conda/envs/reid/Lib/site-packages/certifi/cacert.pem"
ROOT="C:/Users/soura/code/2026/reid/data"

ARCH="$1"
if [ -z "$ARCH" ]; then
    echo "Usage: $0 <arch> [--lr LR] [--bs BS] [--max-epoch N] [--data-fraction F] [--label-smooth] [aug flags...]"
    echo "       $0 <arch> --resume <checkpoint> [--extra N]"
    exit 1
fi
shift

# ---------- defaults ----------
LR=0.0003
OPTIM=amsgrad
BS=64
MAX_EPOCH=30
DATA_FRAC=1.0
RESUME=""
EXTRA=10
LABEL_SMOOTH=0
PASSTHROUGH=()

# ---------- arch-specific defaults (overridden by CLI flags below) ----------
if [ "$ARCH" = "swin_t_fc512" ] || [ "$ARCH" = "deit_s_fc512" ]; then
    OPTIM=sgd
    LR=0.008
fi

# ---------- parse named args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lr)            LR="$2";        shift 2 ;;
        --optim)         OPTIM="$2";     shift 2 ;;
        --bs)            BS="$2";        shift 2 ;;
        --max-epoch)     MAX_EPOCH="$2"; shift 2 ;;
        --data-fraction) DATA_FRAC="$2"; shift 2 ;;
        --resume)        RESUME="$2";    shift 2 ;;
        --extra)         EXTRA="$2";     shift 2 ;;
        --label-smooth)  LABEL_SMOOTH=1; shift ;;
        *)               PASSTHROUGH+=("$1"); shift ;;
    esac
done

# ---------- scheduler ----------
if [ "$ARCH" = "swin_t_fc512" ] || [ "$ARCH" = "deit_s_fc512" ]; then
    SCHEDULER=cosine
    WARMUP_DIV=10
else
    SCHEDULER=multi_step_warmup
    WARMUP_DIV=3
fi

# ---------- derive scheduling values (sets globals) ----------
derive_schedule() {
    local epochs="$1" window="${2:-$1}"
    WARMUP=$(( epochs / WARMUP_DIV )); [ "$WARMUP" -lt 1 ] && WARMUP=1
    EVAL_FREQ=$(( window / 5 ));       [ "$EVAL_FREQ" -lt 1 ] && EVAL_FREQ=1
    PATIENCE=$(( window / 4 ));        [ "$PATIENCE" -lt 1 ] && PATIENCE=1
    if [ "$SCHEDULER" = "multi_step_warmup" ]; then
        STEP1=$(( epochs * 2 / 3 )); [ "$STEP1" -lt 2 ] && STEP1=2
        STEP2=$(( epochs * 5 / 6 )); [ "$STEP2" -le "$STEP1" ] && STEP2=$(( STEP1 + 1 ))
    fi
    return 0
}

# ---------- shared python runner ----------
# run_python <label> <save_dir> [extra python flags...]
run_python() {
    local label="$1" save_dir="$2"; shift 2

    local step_args=()
    [ "$SCHEDULER" = "multi_step_warmup" ] && step_args=(--stepsize "$STEP1" "$STEP2")

    local ls_flag=()
    [ "$LABEL_SMOOTH" = "1" ] && ls_flag=(--label-smooth)

    echo ""
    echo "--- ${label} ---"
    echo "    arch=${ARCH}  lr=${LR}  optim=${OPTIM}  bs=${BS}  epochs=${MAX_EPOCH}  frac=${DATA_FRAC}"
    echo "    scheduler=${SCHEDULER}  warmup=${WARMUP}  eval-freq=${EVAL_FREQ}  patience=${PATIENCE}"
    echo "    save: ${save_dir}"

    STUDENT_ID=ss50456 STUDENT_NAME="Sourav Sen" python main.py \
        -s veri -t veri \
        -a "$ARCH" \
        --experiment "$label" \
        --root "$ROOT" \
        --height 224 --width 224 \
        --optim "$OPTIM" \
        --lr "$LR" \
        --max-epoch "$MAX_EPOCH" \
        "${step_args[@]}" \
        --eval-freq "$EVAL_FREQ" \
        --patience "$PATIENCE" \
        --train-batch-size "$BS" \
        --test-batch-size 100 \
        --workers 4 \
        --data-fraction "$DATA_FRAC" \
        --train-sampler RandomIdentitySampler \
        --num-instances 4 \
        --lr-scheduler "$SCHEDULER" \
        --warmup-epochs "$WARMUP" \
        --save-checkpoint \
        "${ls_flag[@]}" \
        --save-dir "$save_dir" \
        "$@"
}

RUN_STAMP=$(date +%d%m_%H%M)

# ── resume ────────────────────────────────────────────────────────────────────
if [ -n "$RESUME" ]; then
    [ ! -f "$RESUME" ] && { echo "Checkpoint not found: $RESUME"; exit 1; }
    START=$(basename "$RESUME" | sed 's/model\.pth\.tar-//')
    MAX_EPOCH=$(( START + EXTRA ))
    CKPT_DIR=$(dirname "$RESUME")
    _lr=$(basename "$CKPT_DIR" | grep -oE 'lr[0-9.e+-]+' | sed 's/lr//')
    _frac=$(basename "$CKPT_DIR" | grep -oE 'frac[0-9.]+' | sed 's/frac//')
    LR=${_lr:-$LR}
    DATA_FRAC=${_frac:-$DATA_FRAC}
    derive_schedule "$MAX_EPOCH" "$EXTRA"
    echo "Resuming $ARCH from $RESUME (epoch $START → $MAX_EPOCH, extra=$EXTRA)"
    run_python "$ARCH" "$CKPT_DIR" --resume "$RESUME" "${PASSTHROUGH[@]}"
    exit 0
fi

# ── single run ────────────────────────────────────────────────────────────────
derive_schedule "$MAX_EPOCH"
SAVE_DIR="logs/${RUN_STAMP}/train_${ARCH}_lr${LR}_bs${BS}_ep${MAX_EPOCH}_frac${DATA_FRAC}"
run_python "$ARCH" "$SAVE_DIR" "${PASSTHROUGH[@]}"
