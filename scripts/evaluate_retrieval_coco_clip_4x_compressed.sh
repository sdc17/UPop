#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_retrieval_clip.py --evaluate \
--pretrained output/retrieval_coco_clip_compression_4x/clip_large_retrieval_coco_4x_compressed.pth --config ./configs/retrieval_coco_clip.yaml \
--output_dir output/retrieval_coco_clip_compression_4x
