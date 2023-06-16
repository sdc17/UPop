#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_vqa.py --evaluate \
--pretrained output/vqa_vqa2_compression_2x/model_base_vqa_capfilt_large_vqa2_2x_compressed.pth --config ./configs/vqa.yaml \
--output_dir output/vqa_vqa2_compression_2x