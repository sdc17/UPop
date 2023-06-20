#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_retrieval_clip.py --p 0.75 --lr 1e-5 --epoch 18 \
--pretrained pretrained/clip_large_retrieval_flickr.pth --config ./configs/retrieval_flickr_clip.yaml \
--output_dir output/retrieval_flickr_clip_compression_4x
