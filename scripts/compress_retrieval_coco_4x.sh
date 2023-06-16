#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_retrieval.py --p 0.75 --epoch 9 \
--pretrained pretrained/model_base_retrieval_coco.pth --config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco_compression_4x