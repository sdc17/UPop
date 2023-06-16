# UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers

Official implementation of ICML'23 [paper](https://arxiv.org/abs/2301.13741): Unified and Progressive Pruning for Compressing Vision-Language Transformers. The code is tested on Pytorch 1.11.0, and the dependencies can be installed by <pre/> conda install --yes --file requirements.txt </pre>



### Visual Reasoning on NLVR2

* Dataset

    Download NLVR2 dataset, unzip it under `datasets` folder, and accordingly modify `image_root` in [config](./configs/nlvr.yaml).

* Annotation
  
Download all-in-one annonations from [download](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under `annotation` folder, and accordingy modify `annotation` in [config](./configs/nlvr.yaml).

* Evaluate
  
Download checkpoints from the table below, put thme under `output` folder, and accordingly modify `--output` of the scripts. For example, to evaluate a 2x compressed model:
<pre/>
python -m torch.distributed.run --nproc_per_node=8 compress_nlvr.py --evaluate \
--pretrained output/nlvr_nlvr2_compression_2x/model_base_nlvr_nlvr2_2x_compressed.pth --config ./configs/nlvr.yaml \
--output_dir output/nlvr_nlvr2_compression_2x
</pre>

* Compress
  
Download the uncompressed model from the table below, put it under `pretrained` folder, and accodingly modify `pretrained` in [config](./configs/nlvr.yaml). For example, to conduct a 2x compression:
<pre/>
python -m torch.distributed.run --nproc_per_node=8 compress_nlvr.py --p 0.5 --epoch 15 \
--pretrained pretrained/model_base_nlvr.pth --config ./configs/nlvr.yaml \
--output_dir output/nlvr_nlvr2_compression_2x
</pre>


### Acknowledgement
This code is bulit upon <a href="https://github.com/salesforce/BLIP">BLIP</a>. We thank the original authors for their open source work.


### Citation
If you find this work useful, please consider citing the corresponding paper:
<pre/>
@article{shi2023upop,
  title={Upop: Unified and progressive pruning for compressing vision-language transformers},
  author={Shi, Dachuan and Tao, Chaofan and Jin, Ying and Yang, Zhendong and Yuan, Chun and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2301.13741},
  year={2023}
}
</pre>


