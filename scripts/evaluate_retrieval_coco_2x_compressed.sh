#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_retrieval.py --evaluate \
--pretrained output/retrieval_coco_compression_2x/model_base_retrieval_coco_2x_compressed.pth --config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco_compression_2x