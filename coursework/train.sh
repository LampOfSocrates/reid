#!/bin/bash
# Interactive launcher for main.py.
# Assumes --root is correct and only prompts for model + key hyperparameters.

set -e

# Always run from the coursework dir so main.py resolves regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ROOT="C:/Users/soura/code/2026/reid/data"

MODELS=(
    "resnet18"
    "resnet18_fc512"
    "resnet34"
    "resnet34_fc512"
    "resnet50"
    "resnet50_fc512"
    "mobilenet_v3_small"
    "swin_t_fc512"
    "swin_t_custom"
    "clip_senet_v1_dual"
    "clip_senet_v2_frozen"
    "clip_senet_v3_ibn_supcon"
)

# Defaults (match the previous train.sh).
DEFAULT_SOURCE="veri"
DEFAULT_TARGET="veri"
DEFAULT_HEIGHT=224
DEFAULT_WIDTH=224
DEFAULT_OPTIM="amsgrad"
DEFAULT_LR=0.0003
DEFAULT_MAX_EPOCH=10
DEFAULT_STEPSIZE="20 40"
DEFAULT_TRAIN_BS=64
DEFAULT_TEST_BS=100
DEFAULT_DATA_FRACTION=0.01

prompt() {
    # prompt VAR_NAME "Label" "default"
    local var="$1" label="$2" default="$3" reply
    read -r -p "  ${label} [${default}]: " reply
    printf -v "$var" '%s' "${reply:-$default}"
}

echo "=========================================="
echo " Vehicle ReID training launcher"
echo "=========================================="
echo "Root (fixed):       ${ROOT}"
echo
echo "Available models:"
for i in "${!MODELS[@]}"; do
    printf "  %2d) %s\n" "$((i + 1))" "${MODELS[$i]}"
done
echo

while :; do
    read -r -p "Select model [1-${#MODELS[@]}] (default 7 = mobilenet_v3_small): " choice
    choice="${choice:-7}"
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#MODELS[@]} )); then
        ARCH="${MODELS[$((choice - 1))]}"
        break
    fi
    echo "  invalid choice, try again."
done
echo "  -> ${ARCH}"
echo

echo "Hyperparameters (press Enter to accept default):"
prompt SOURCE      "source dataset (-s)"     "$DEFAULT_SOURCE"
prompt TARGET      "target dataset (-t)"     "$DEFAULT_TARGET"
prompt HEIGHT      "image height"            "$DEFAULT_HEIGHT"
prompt WIDTH       "image width"             "$DEFAULT_WIDTH"
prompt OPTIM       "optimizer"               "$DEFAULT_OPTIM"
prompt LR          "learning rate"           "$DEFAULT_LR"
prompt MAX_EPOCH   "max epochs"              "$DEFAULT_MAX_EPOCH"
prompt STEPSIZE    "lr stepsize (space-sep)" "$DEFAULT_STEPSIZE"
prompt TRAIN_BS    "train batch size"        "$DEFAULT_TRAIN_BS"
prompt TEST_BS     "test batch size"         "$DEFAULT_TEST_BS"
prompt DATA_FRAC   "data fraction (0-1)"     "$DEFAULT_DATA_FRACTION"

SAVE_DIR="logs/${ARCH}-${SOURCE}"

echo
echo "------------------------------------------"
echo " Run summary"
echo "------------------------------------------"
printf "  arch         : %s\n" "$ARCH"
printf "  root         : %s\n" "$ROOT"
printf "  source/target: %s / %s\n" "$SOURCE" "$TARGET"
printf "  size (HxW)   : %s x %s\n" "$HEIGHT" "$WIDTH"
printf "  optim/lr     : %s @ %s\n" "$OPTIM" "$LR"
printf "  epochs       : %s (stepsize: %s)\n" "$MAX_EPOCH" "$STEPSIZE"
printf "  batch (tr/te): %s / %s\n" "$TRAIN_BS" "$TEST_BS"
printf "  data-fraction: %s\n" "$DATA_FRAC"
printf "  save-dir     : %s\n" "$SAVE_DIR"
echo "------------------------------------------"

read -r -p "Proceed? [Y/n] " go
case "${go:-Y}" in
    [Yy]*) ;;
    *) echo "Aborted."; exit 0 ;;
esac

STUDENT_ID=ss50456 STUDENT_NAME="Sourav Sen" python main.py \
    -s "$SOURCE" \
    -t "$TARGET" \
    -a "$ARCH" \
    --root "$ROOT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --optim "$OPTIM" \
    --lr "$LR" \
    --max-epoch "$MAX_EPOCH" \
    --stepsize $STEPSIZE \
    --train-batch-size "$TRAIN_BS" \
    --test-batch-size "$TEST_BS" \
    --data-fraction "$DATA_FRAC" \
    --save-dir "$SAVE_DIR"
