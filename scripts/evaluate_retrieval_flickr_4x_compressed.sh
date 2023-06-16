#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_retrieval_flickr.py --evaluate \
--pretrained output/retrieval_flickr_compression_4x/model_base_retrieval_flickr_4x_compressed.pth --config ./configs/retrieval_flickr.yaml \
--output_dir output/retrieval_flickr_compression_4x
