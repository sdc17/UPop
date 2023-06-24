#!/bin/bash

export DATASET=datasets/vision

# for single-scale testing
python -m torch.distributed.run --nproc_per_node=4 --master_port 10603 segm/eval/miou.py \
output/seg_small_mask_16s_64r_020x/seg_small_mask_020x_compressed.pth ade20k --singlescale

# for multi-scale testing
python -m torch.distributed.run --nproc_per_node=4 --master_port 10603 segm/eval/miou.py \
output/seg_small_mask_16s_64r_020x/seg_small_mask_020x_compressed.pth ade20k --multiscale
