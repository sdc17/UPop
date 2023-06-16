# UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers

Official implementation of ICML'23 [paper](https://arxiv.org/abs/2301.13741): Unified and Progressive Pruning for Compressing Vision-Language Transformers. The code is tested on Pytorch 1.11.0, and the dependencies can be installed by <pre/> conda install --yes --file requirements.txt </pre>



### Visual Reasoning on NLVR2

* Dataset & Annotation

    Download NLVR2 dataset, unzip it under `datasets` folder, and accordingly modify `image_root` in [config](./configs/nlvr.yaml). Download all-in-one annonations from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under `annotation` folder, and accordingy modify `annotation` in [config](./configs/nlvr.yaml).

* Evaluation
  
    Download compresssed checkpoints from the table below, put them under `output` folder, and accordingly modify `--output` of the scripts. For example, to evaluate a 2x compressed model: 
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_nlvr.py --evaluate \
    --pretrained output/nlvr_nlvr2_compression_2x/model_base_nlvr_nlvr2_2x_compressed.pth --config ./configs/nlvr.yaml \
    --output_dir output/nlvr_nlvr2_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under `pretrained` folder, and accodingly modify `pretrained` in [config](./configs/nlvr.yaml). For example, to conduct a 2x compression on 8 A100 GPUs:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_nlvr.py --p 0.5 --epoch 15 \
    --pretrained pretrained/model_base_nlvr.pth --config ./configs/nlvr.yaml \
    --output_dir output/nlvr_nlvr2_compression_2x
    ```

* Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    2x | <a href="https://drive.google.com/uc?export=download&id=1pcsvlNRzzoq_q6Kaku_Kkg1MFELGoIxE">Download</a> | [Link](./scripts/compress_nlvr_nlvr2_2x.sh) | <a href="https://drive.google.com/uc?export=download&id=10Olyj0IBji3t2QwL85FK-gwjfm87k7tT">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1HIh45vjaNUSy20uPSg_rtXtAlV9w7VB4">Download</a> | [Link](./scripts/evaluate_nlvr_nlvr2_2x_compressed.sh)
    3x | <a href="https://drive.google.com/uc?export=download&id=1pcsvlNRzzoq_q6Kaku_Kkg1MFELGoIxE">Download</a> | [Link](./scripts/compress_nlvr_nlvr2_3x.sh) | <a href="https://drive.google.com/uc?export=download&id=1amXIX9bXMiWSopkHRbVUHHqJDfkBxRie">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1fdCW-HrsPrqpHpCCvypgKTOePjPdtKOc">Download</a> | [Link](./scripts/evaluate_nlvr_nlvr2_3x_compressed.sh)
    4x | <a href="https://drive.google.com/uc?export=download&id=1pcsvlNRzzoq_q6Kaku_Kkg1MFELGoIxE">Download</a> | [Link](./scripts/compress_nlvr_nlvr2_4x.sh)| <a href="https://drive.google.com/uc?export=download&id=1bSqP6ODlUiT24ACN3nN0HuGx341YJI9B">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1OaHw4Bn1Kp1EfR6L3f1Xby-i2MO-_oJs">Download</a> | [Link](./scripts/evaluate_nlvr_nlvr2_4x_compressed.sh)
    5x | <a href="https://drive.google.com/uc?export=download&id=1pcsvlNRzzoq_q6Kaku_Kkg1MFELGoIxE">Download</a> | [Link](./scripts/compress_nlvr_nlvr2_5x.sh)| <a href="https://drive.google.com/uc?export=download&id=16rmyQ1sGZma5_VoXT6ew5T72053wxwQ3">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1eAvTeJH8EOvjJMwpFw66tPsOPodPZT2a">Download</a> | [Link](./scripts/evaluate_nlvr_nlvr2_5x_compressed.sh)
    10x | <a href="https://drive.google.com/uc?export=download&id=1pcsvlNRzzoq_q6Kaku_Kkg1MFELGoIxE">Download</a> | [Link](./scripts/compress_nlvr_nlvr2_10x.sh) | <a href="https://drive.google.com/uc?export=download&id=1g4FRQKkrn_8zPGLdN0S1-AQaCasUuWeR">Download</a> | <a href="https://drive.google.com/uc?export=download&id=12Exrv25avoZxXrmx4JhWMVXfRUpu-qSE">Download</a> | [Link](./scripts/evaluate_nlvr_nlvr2_10x_compressed.sh)

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


