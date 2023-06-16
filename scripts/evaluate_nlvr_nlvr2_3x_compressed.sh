#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_nlvr.py --evaluate \
--pretrained output/nlvr_nlvr2_compression_3x/model_base_nlvr_nlvr2_3x_compressed.pth --config ./configs/nlvr.yaml \
--output_dir output/nlvr_nlvr2_compression_3x