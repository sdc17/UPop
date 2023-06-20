#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_retrieval_clip.py --p 0.5 --lr 1e-5 --epoch 6 \
--pretrained pretrained/clip_large_retrieval_coco.pth --config ./configs/retrieval_coco_clip.yaml \
--output_dir output/retrieval_coco_clip_compression_2x