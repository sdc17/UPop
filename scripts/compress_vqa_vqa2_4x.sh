#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_vqa.py --p 0.75 --epoch 15 \
--pretrained pretrained/model_base_vqa_capfilt_large.pth --config ./configs/vqa.yaml \
--output_dir output/vqa_vqa2_compression_4x
