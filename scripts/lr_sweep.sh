#!/bin/bash
cd "$(dirname "$0")/.."

./scripts/train.sh swin_t_fc512 --lr 0.0003 --bs 64 --max-epoch 30 --data-fraction 1.0 > train_swin_t_fc512_0p0003_64_30_1.out 2>&1 ;
./scripts/train.sh swin_t_fc512 --lr 0.003  --bs 64 --max-epoch 30 --data-fraction 1.0 > train_swin_t_fc512_0p003_64_30_1.out  2>&1 ;
./scripts/train.sh swin_t_fc512 --lr 0.008  --bs 64 --max-epoch 30 --data-fraction 1.0 > train_swin_t_fc512_0p008_64_30_1.out  2>&1 ;
./scripts/train.sh swin_t_fc512 --lr 0.01   --bs 64 --max-epoch 30 --data-fraction 1.0 > train_swin_t_fc512_0p01_64_30_1.out   2>&1 ;
./scripts/train.sh swin_t_fc512 --lr 0.05   --bs 64 --max-epoch 30 --data-fraction 1.0 > train_swin_t_fc512_0p05_64_30_1.out   2>&1


echo "RUNNING LR SWEEP with BS = 64" ; 
./scripts/train.sh swin_t_fc512 --lr 0.008 --bs 32  --max-epoch 30 --data-fraction 1.0 > train_swin_t_fc512_0p008_32_30_1.out 2>&1 ; 
./scripts/train.sh swin_t_fc512 --lr 0.008 --bs 48  --max-epoch 30 --data-fraction 1.0 > train_swin_t_fc512_0p008_48_30_1.out 2>&1 ; 
./scripts/train.sh swin_t_fc512 --lr 0.008 --bs 64  --max-epoch 30 --data-fraction 1.0 > train_swin_t_fc512_0p008_64_30_1.out 2>&1 ; 
./scripts/train.sh swin_t_fc512 --lr 0.008 --bs 96  --max-epoch 30 --data-fraction 1.0 > train_swin_t_fc512_0p008_96_30_1.out 2>&1 ;
./scripts/train.sh swin_t_fc512 --lr 0.008 --bs 128 --max-epoch 30 --data-fraction 1.0 > train_swin_t_fc512_0p008_128_30_1.out 2>&1