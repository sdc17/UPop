#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_caption.py --p 0.5 --epoch 5 \
--pretrained pretrained/model_base_caption_capfilt_large.pth --config ./configs/caption_coco.yaml \
--output_dir output/caption_coco_compression_2x