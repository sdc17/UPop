#!/usr/bin/env sh
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_deit.py \
--data-path datasets/vision/imagenet \
--finetune pretrained/deit_small_patch16_224-cd65a155.pth \
--model deit_small_patch16_224 \
--epochs-search 60 \
--epochs 300 \
--batch-size 512 \
--lr-search 1e-4 \
--lr 1e-4 \
--warmup-epochs 0 \
--p 0.4 \
--interval 800 \
--output_dir output/train_deit_small_patch16_224_60s_300r_040x