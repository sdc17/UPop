#!/bin/bash

export DATASET=datasets/vision

python -m torch.distributed.run --nproc_per_node=4 --master_port 10603 segm/train.py --dataset ade20k \
  --backbone vit_small_patch16_384 --decoder mask_transformer --no-resume \
  --pretrained pretrained/seg_small_mask.pth \
  --epochs-search 16 \
  --epochs 64 \
  --batch-size 64 \
  --lr-search 4e-3 \
  -lr 4e-3  \
  --p 0.20 \
  --interval 300 \
  --log-dir output/seg_small_mask_16s_64r_020x
