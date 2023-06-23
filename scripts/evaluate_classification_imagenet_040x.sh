#!/usr/bin/env sh
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_deit.py --eval --dist-eval \
--data-path datasets/vision/imagenet \
--model deit_small_patch16_224 \
--resume output/train_deit_small_patch16_224_60s_300r_040x/deit_small_patch16_224_040x_compressed.pth