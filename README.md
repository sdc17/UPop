# UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers


<p align="center">
    <a href="https://github.com/sdc17/UPop/actions/workflows/build.yml">
        <img alt="Build" src="https://github.com/sdc17/UPop/actions/workflows/build.yml/badge.svg" />
    </a>
    <a href="https://proceedings.mlr.press/v202/shi23e/shi23e.pdf">
        <img alt="Paper" src="https://img.shields.io/badge/paper-link-blue?logo=quicklook" />
    </a>
    <a href="https://arxiv.org/abs/2301.13741">
        <img alt="Paper" src="https://img.shields.io/badge/arXiv-link-B31B1B?logo=arxiv" />
    </a>
    <a href="https://github.com/sdc17/UPop">
        <img alt="Code" src="https://img.shields.io/badge/code-link-181717?logo=github" />
    </a>
    <a href="https://dachuanshi.com/UPop-Project/">
        <img alt="Webiste" src="https://img.shields.io/badge/website-link-4285F4?logo=googleearth" />
    </a>
    <a href="https://dachuanshi.medium.com/compressing-multimodal-and-unimodal-transformers-via-upop-466c11680ac0">
        <img alt="Blog" src="https://img.shields.io/badge/blog-in English-FFA500?logo=rss" />
    </a>
    <a href="https://zhuanlan.zhihu.com/p/640634482">
        <img alt="Blog" src="https://img.shields.io/badge/blog-中文-FFA500?logo=rss" />
    </a><br>
    <a href="https://pytorch.org/get-started/previous-versions/">
        <img alt="Pytorch" src="https://img.shields.io/badge/pytorch-v1.11.0-EE4C2C?logo=pytorch" />
    </a>
    <a href="https://www.python.org/downloads/release/python-3811/">
        <img alt="Pytorch" src="https://img.shields.io/badge/python-v3.8.11-3776AB?logo=python" />
    </a>
    <a href="https://github.com/sdc17/UPop/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-BSD 3--Clause-F96702?logo=cloudera&logoColor=c0c0c0" />
    </a>
</p>
<!-- <img src="UPop.png" width="800"> -->

### 🧐 A Quick Look 

* **What is it**: UPop is the first **structured pruning framework** for vision-language Transformers. It **enables effective structured pruning on various multi-modal & uni-modal tasks** (including Visual Reasoning, Image Captioning, Visual Question Answer, Image-Text Retrieval, Text-Image Retrieval, Image Classification and Image Segmentation), **datasets** (including NLVR2, COCO Caption, VQAv2, COCO, Flickr30K, ImageNet and ADE20K), and **model architectures** (including BLIP, CLIP, DeiT and Segmenter).

    https://github.com/sdc17/UPop-Project/assets/47023705/7561f7a3-8f5c-4ab6-88b1-30dda533f3fe

* **What challenge does it tackle**: The above video demonstrates that **Unified Search** adopted by UPop **rescues us from the burden of repeated experiments** (e.g., doing grid search) for searching optimal compression ratios among different modalities and structures. Furthermore, **Progressive Pruning** adopted by UPop eliminates the weight gap between the searched model and the pruned subnet to be retrained, therefore **gaining better convergence and performance**, especially at high compression ratios.

* **How about the performance**: On multimodal tasks, for example, UPop can achieve **2x compression with only 1.2% and 2.0% accuracy loss on the VQAv2 dataset for Visual Question Answer and the NLVR2 dataset for Visual Reasoning**, respectively. On unimodal tasks, for example, UPop can achieve **1.5x and 1.2x compression without any loss of accuracy on the ImageNet dataset for Image Classification and the ADE20K dataset for Image Segmentation**, respectively. Some examples of **vector-level structured** granularity are as follows.

    Example (Task • Dataset • Model • Metric) | Performance | Parameters (M) | FLOPs (G) 
    --- | --- | --- | ---
    [Visual Reasoning](https://github.com/sdc17/UPop#-visual-reasoning-on-the-nlvr2-dataset) • [NLVR2](https://lil.nlp.cornell.edu/nlvr/) • [BLIP](https://github.com/salesforce/BLIP) • Acc | $83.1 \rightarrow 81.1_{\color{red}\downarrow 2.0}$ | $259.5 \rightarrow 150.2_{\color{ForestGreen}\downarrow 42\\%}$ |  $132.5 \rightarrow 89.4_{\color{ForestGreen}\downarrow 33\\%}$ 
    [Image Caption](https://github.com/sdc17/UPop#-image-caption-on-the-coco-caption-dataset) • [Caption COCO](https://cocodataset.org/#home) • [BLIP](https://github.com/salesforce/BLIP) • SPICE|  $23.8 \rightarrow 23.3_{\color{red}\downarrow 0.5}$ | $224.0 \rightarrow 127.1_{\color{ForestGreen}\downarrow 43\\%}$ | $65.7 \rightarrow 39.8_{\color{ForestGreen}\downarrow 39\\%}$
    [Visual Question Answer](https://github.com/sdc17/UPop#-visual-question-answer-on-the-vqav2-dataset) • [VQAv2](https://visualqa.org/) • [BLIP](https://github.com/salesforce/BLIP) • Acc | $77.5 \rightarrow 76.3_{\color{red}\downarrow 1.2}$ | $361.6 \rightarrow 211.3_{\color{ForestGreen}\downarrow 42\\%}$ | $186.1 \rightarrow 109.4_{\color{ForestGreen}\downarrow 41\\%}$
    [Image-Text Retrieval](https://github.com/sdc17/UPop#-image-text-and-text-image-retrieval-on-the-coco-dataset) • [COCO](https://cocodataset.org/#home) • [BLIP](https://github.com/salesforce/BLIP) • R@1 | $81.9 \rightarrow 77.4_{\color{red}\downarrow 4.5}$ | $447.6 \rightarrow 248.9_{\color{ForestGreen}\downarrow 44\\%}$ | $153.2\rightarrow 88.3_{\color{ForestGreen}\downarrow 42\\%}$ 
    [Image-Text Retrieval](https://github.com/sdc17/UPop#-image-text-and-text-image-retrieval-on-the-coco-dataset-with-clip) • [COCO](https://cocodataset.org/#home) • [CLIP](https://github.com/openai/CLIP) • R@1 | $71.5 \rightarrow 70.8_{\color{red}\downarrow 0.7}$ | $856.0 \rightarrow 473.7_{\color{ForestGreen}\downarrow 45\\%}$ | $395.7\rightarrow 196.3_{\color{ForestGreen}\downarrow 50\\%}$ 
    [Text-Image Retrieval](https://github.com/sdc17/UPop#-image-text-and-text-image-retrieval-on-the-coco-dataset) • [COCO](https://cocodataset.org/#home) • [BLIP](https://github.com/salesforce/BLIP) • R@1 | $64.3\rightarrow 59.8_{\color{red}\downarrow 4.5}$ | $447.6 \rightarrow 248.9_{\color{ForestGreen}\downarrow 44\\%}$ | $153.2\rightarrow 88.3_{\color{ForestGreen}\downarrow 42\\%}$ 
    [Text-Image Retrieval](https://github.com/sdc17/UPop#-image-text-and-text-image-retrieval-on-the-coco-dataset-with-clip) • [COCO](https://cocodataset.org/#home) • [CLIP](https://github.com/openai/CLIP) • R@1 | $56.8\rightarrow 53.1_{\color{red}\downarrow 3.7}$ | $856.0 \rightarrow 473.7_{\color{ForestGreen}\downarrow 45\\%}$ | $395.7\rightarrow 196.3_{\color{ForestGreen}\downarrow 50\\%}$ 
    [Image-Text Retrieval](https://github.com/sdc17/UPop#-image-text-and-text-image-retrieval-on-the-flickr30k-dataset) • [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) • [BLIP](https://github.com/salesforce/BLIP) • R@1 |  $96.8\rightarrow 92.2_{\color{red}\downarrow 4.4}$ | $447.6\rightarrow 250.5_{\color{ForestGreen}\downarrow 44\\%}$ | $153.2\rightarrow 91.0_{\color{ForestGreen}\downarrow 41\\%}$ 
    [Image-Text Retrieval](https://github.com/sdc17/UPop#-image-text-and-text-image-retrieval-on-the-flickr30k-dataset-with-clip) • [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) • [CLIP](https://github.com/openai/CLIP) • R@1 |  $96.8\rightarrow 93.2_{\color{red}\downarrow 3.6}$ | $856.0\rightarrow 474.3_{\color{ForestGreen}\downarrow 45\\%}$ | $395.7 \rightarrow 201.1_{\color{ForestGreen}\downarrow 49\\%}$ 
    [Text-Image Retrieval](https://github.com/sdc17/UPop#-image-text-and-text-image-retrieval-on-the-flickr30k-dataset) • [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) • [BLIP](https://github.com/salesforce/BLIP) • R@1 |  $86.9 \rightarrow 82.0_{\color{red}\downarrow 4.9}$ | $447.6\rightarrow 250.5_{\color{ForestGreen}\downarrow 44\\%}$ | $153.2\rightarrow 91.0_{\color{ForestGreen}\downarrow 41\\%}$ 
    [Text-Image Retrieval](https://github.com/sdc17/UPop#-image-text-and-text-image-retrieval-on-the-flickr30k-dataset-with-clip) • [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) • [CLIP](https://github.com/openai/CLIP) • R@1 |  $86.6\rightarrow 80.5_{\color{red}\downarrow 6.1}$ | $856.0\rightarrow 474.3_{\color{ForestGreen}\downarrow 45\\%}$ | $395.7 \rightarrow 201.1_{\color{ForestGreen}\downarrow 49\\%}$ 
    [Classification](https://github.com/sdc17/UPop#-image-classification-on-the-imagenet-dataset) • [ImageNet](https://www.image-net.org/) • [DeiT](https://github.com/facebookresearch/deit) • Acc@1 | $79.9\rightarrow 80.2_{\color{ForestGreen}\uparrow 0.3}$ | $22.0 \rightarrow 15.7_{\color{ForestGreen}\downarrow 29\\%}$ | $4.6 \rightarrow 3.2_{\color{ForestGreen}\downarrow 30\\%}$
    [Classification](https://github.com/sdc17/UPop#-image-classification-on-the-imagenet-dataset) • [ImageNet](https://www.image-net.org/) • [DeiT](https://github.com/facebookresearch/deit) • Acc@5 |  $95.0 \rightarrow 95.1_{\color{ForestGreen}\uparrow 0.1}$ | $22.0 \rightarrow 15.7_{\color{ForestGreen}\downarrow 29\\%}$ | $4.6 \rightarrow 3.2_{\color{ForestGreen}\downarrow 30\\%}$ 
    [Segmentation](https://github.com/sdc17/UPop#-image-classification-on-the-imagenet-dataset) • [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) • [Segmenter](https://github.com/rstrudel/segmenter) • $\text{mIoU}^s$ | $45.3\rightarrow 45.3_{\color{ForestGreen}\uparrow 0.0}$ | $26.4 \rightarrow 21.5_{\color{ForestGreen}\downarrow 19\\%}$ | $38.6 \rightarrow 30.4_{\color{ForestGreen}\downarrow 21\\%}$ 
    [Segmentation](https://github.com/sdc17/UPop#-image-classification-on-the-imagenet-dataset) • [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K) • [Segmenter](https://github.com/rstrudel/segmenter) • $\text{mIoU}^m$ | $46.9 \rightarrow 47.1_{\color{ForestGreen}\uparrow 0.2}$ | $26.4 \rightarrow 21.5_{\color{ForestGreen}\downarrow 19\\%}$ | $38.6 \rightarrow 30.4_{\color{ForestGreen}\downarrow 21\\%}$ 

### 🥳 What's New 
* (Jun 2023), we worked on a new project CrossGET: Cross-Guided Ensemble of Tokens for Accelerating Vision-Language Transformers, which reduces computational costs effectively for accelerating. [[Paper]](https://arxiv.org/pdf/2305.17455.pdf) [[Code]](https://github.com/sdc17/CrossGET) 💡

* (Jun 30, 2023), we released the ```implementation```, ```scripts```, ```checkpoints```, and ```logs```. [[Code]](https://github.com/sdc17/UPop) [[Website]](https://dachuanshi.com/UPop-Project/) 🚩

* (Apr 25, 2023), our work UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers was accepted by ICML 2023. [[Paper]](https://proceedings.mlr.press/v202/shi23e/shi23e.pdf) [[ArXiv]](https://arxiv.org/abs/2301.13741) 🎉


### 🏃 Installation
The code is tested on `Pytorch==1.11.0`, `cuda==11.3.1`, and `python==3.8.13`. The dependencies can be installed by:
```
conda env create -f environment.yml
```
The status of installing dependencies: [![build](https://github.com/sdc17/UPop/actions/workflows/build.yml/badge.svg)](https://github.com/sdc17/UPop/actions/workflows/build.yml)

<!-- ### Supported Tasks, Models, and Datasets
Type |  Supported Tasks | Supported Models  | Supported Datasets |
--- | --- | :---: | :---: 
Multi-modal | [Visual Reasoning](https://github.com/sdc17/UPop#visual-reasoning-on-the-nlvr2-dataset) | [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/sdc17/UPop#visual-reasoning-on-the-nlvr2-dataset)) | [NLVR2](https://lil.nlp.cornell.edu/nlvr/)
Multi-modal |[Image Caption](https://github.com/sdc17/UPop#image-caption-on-the-coco-caption-dataset) | [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/sdc17/UPop#image-caption-on-the-coco-caption-dataset)) | [COCO Caption](https://cocodataset.org/#home)
Multi-modal |[Visual Question Answer](https://github.com/sdc17/UPop#visual-question-answer-on-the-vqav2-dataset) | [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/sdc17/UPop#visual-question-answer-on-the-vqav2-dataset)) | [VQAv2](https://visualqa.org/)
Multi-modal |[Image-Text Retrieval](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-coco-dataset) | [CLIP](https://github.com/openai/CLIP) ([instructions](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-coco-dataset-with-clip)), [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-coco-dataset)) | [COCO](https://cocodataset.org/#home), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)
Multi-modal |[Text-Image Retrieval](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-coco-dataset) | [CLIP](https://github.com/openai/CLIP) ([instructions](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-flickr30k-dataset-with-clip)), [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-flickr30k-dataset)) | [COCO](https://cocodataset.org/#home), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)
Uni-modal |[Image Classification](https://github.com/sdc17/UPop#image-classification-on-the-imagenet-dataset) | [DeiT](https://github.com/facebookresearch/deit) ([instructions](https://github.com/sdc17/UPop#image-classification-on-the-imagenet-dataset)) | [ImageNet](https://www.image-net.org/)
Uni-modal |[Image Segmentation](https://github.com/sdc17/UPop#image-segmentation-on-the-ade20k-dataset) | [Segmenter](https://github.com/rstrudel/segmenter) ([instructions](https://github.com/sdc17/UPop#image-segmentation-on-the-ade20k-dataset)) | [Ade20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/) -->

### 🚀 Visual Reasoning on the NLVR2 Dataset

* Dataset & Annotation

    Download the [NLVR2](https://lil.nlp.cornell.edu/nlvr/) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/nlvr.yaml). Download all-in-one annotations (including annotations for Visual Reasoning, Image Caption, VQA, Image-Text Retrieval, and Text-Image Retrieval tasks) from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/nlvr.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_nlvr.py --evaluate \
    --pretrained output/nlvr_nlvr2_compression_2x/model_base_nlvr_nlvr2_2x_compressed.pth \
    --config ./configs/nlvr.yaml \
    --output_dir output/nlvr_nlvr2_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/nlvr.yaml). For example, to conduct a 2x compression on 8 A100 GPUs:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_nlvr.py --p 0.5 --epoch 15 \
    --pretrained pretrained/model_base_nlvr.pth \
    --config ./configs/nlvr.yaml \
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



### 🚀 Image Caption on the COCO Caption Dataset

* Dataset & Annotation

    Download the [COCO Caption](https://cocodataset.org/#home) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/caption_coco.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/caption_coco.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_caption.py --evaluate \
    --pretrained output/caption_coco_compression_2x/model_base_caption_capfilt_large_coco_2x_compressed.pth \
    --config ./configs/caption_coco.yaml \
    --output_dir output/caption_coco_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/caption_coco.yaml). For example, to conduct a 2x compression on 8 A100 GPUs:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_caption.py --p 0.5 --epoch 5 \
    --pretrained pretrained/model_base_caption_capfilt_large.pth \
    --config ./configs/caption_coco.yaml \
    --output_dir output/caption_coco_compression_2x
    ```

* Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    2x | <a href="https://drive.google.com/uc?export=download&id=1qW_0DpQsDc6u9g3fSfTI4g_VXYsMA5s8">Download</a> | [Link](./scripts/compress_caption_coco_2x.sh) | <a href="https://drive.google.com/uc?export=download&id=1LkaQ1xGEdUvoVo_frvHT5rJDR6TqaoMr">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1GoztmYFYNsU0FdsTs3_mYfx4t1fS7o6E">Download</a> | [Link](./scripts/evaluate_caption_coco_2x_compressed.sh)
    4x | <a href="https://drive.google.com/uc?export=download&id=1qW_0DpQsDc6u9g3fSfTI4g_VXYsMA5s8">Download</a> | [Link](./scripts/compress_caption_coco_4x.sh)| <a href="https://drive.google.com/uc?export=download&id=1kPggFkmnikveSn20dKXOjTrJbakaZBAl">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1Pp947bX2yVApghKCJi5DJB0YsDGDgiTz">Download</a> | [Link](./scripts/evaluate_caption_coco_4x_compressed.sh)
    


### 🚀 Visual Question Answer on the VQAv2 Dataset

* Dataset & Annotation

    Download the [VQAv2](https://visualqa.org/) dataset and [Visual Genome](https://visualgenome.org/) dataset, unzip them under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/vqa.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/vqa.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model: (Note that the scripts will generate answers `vqa_result.json`, which should be submitted to the [official server](https://eval.ai/web/challenges/challenge-page/830/overview) to obtain evaluation results.) 
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_vqa.py --evaluate \
    --pretrained output/vqa_vqa2_compression_2x/model_base_vqa_capfilt_large_vqa2_2x_compressed.pth \
    --config ./configs/vqa.yaml \
    --output_dir output/vqa_vqa2_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/vqa.yaml). For example, to conduct a 2x compression on 8 A100 GPUs:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_vqa.py --p 0.5 --epoch 10 \
    --pretrained pretrained/model_base_vqa_capfilt_large.pth \
    --config ./configs/vqa.yaml \
    --output_dir output/vqa_vqa2_compression_2x
    ```

* Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    2x | <a href="https://drive.google.com/uc?export=download&id=18Ihg2NA_puj3_92uVszqonSusLFgmID-">Download</a> | [Link](./scripts/compress_vqa_vqa2_2x.sh) | <a href="https://drive.google.com/uc?export=download&id=1Qmv73cTvzU3AQbUer9Jjekebzc4U175C">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1K1OAD2y2h8WHYwp6r0A4WRcS654FfKYr">Download</a> | [Link](./scripts/evaluate_vqa_vqa2_2x_compressed.sh)
    4x | <a href="https://drive.google.com/uc?export=download&id=18Ihg2NA_puj3_92uVszqonSusLFgmID-">Download</a> | [Link](./scripts/compress_vqa_vqa2_4x.sh)| <a href="https://drive.google.com/uc?export=download&id=1_VDsABugk9LNt9mMUD5Z_BuO4Ir9V2_k">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1abiAFOZtK64HSMe9JHffwY7e_7M86PJU">Download</a> | [Link](./scripts/evaluate_vqa_vqa2_4x_compressed.sh)
    

### 🚀 Image-Text and Text-Image Retrieval on the COCO Dataset

* Dataset & Annotation

    Download the [COCO](https://cocodataset.org/#home) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_coco.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_coco.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval.py --evaluate \
    --pretrained output/retrieval_coco_compression_2x/model_base_retrieval_coco_2x_compressed.pth --config ./configs/retrieval_coco.yaml \
    --output_dir output/retrieval_coco_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_coco.yaml). For example, to conduct a 2x compression on 8 A100 GPUs:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval.py --p 0.5 --epoch 6 \
    --pretrained pretrained/model_base_retrieval_coco.pth \
    --config ./configs/retrieval_coco.yaml \
    --output_dir output/retrieval_coco_compression_2x
    ```

* Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    2x | <a href="https://drive.google.com/uc?export=download&id=19nxvphpnIH2kbV4unL0MDAM_2zlBnruq">Download</a> | [Link](./scripts/compress_retrieval_coco_2x.sh) | <a href="https://drive.google.com/uc?export=download&id=1EACJZO2QdbcLkBr6uvZq_z8A4vXzc8yK">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1tDo3gk4IQUHgm21RbK96Qg-haEFBAgBX">Download</a> | [Link](./scripts/evaluate_retrieval_coco_2x_compressed.sh)
    4x | <a href="https://drive.google.com/uc?export=download&id=19nxvphpnIH2kbV4unL0MDAM_2zlBnruq">Download</a> | [Link](./scripts/compress_retrieval_coco_4x.sh)| <a href="https://drive.google.com/uc?export=download&id=1-OA-xkLbzH39GPfrVFux3wNZ9h0GwyJX">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1G5FFff4r5lT0WhUXmxfO8nOUtNgwD_PY">Download</a> | [Link](./scripts/evaluate_retrieval_coco_4x_compressed.sh)
    

### 🚀 Image-Text and Text-Image Retrieval on the Flickr30K Dataset

* Dataset & Annotation

    Download the [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_flickr.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_flickr.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_flickr.py --evaluate \
    --pretrained output/retrieval_flickr_compression_2x/model_base_retrieval_flickr_2x_compressed.pth \
    --config ./configs/retrieval_flickr.yaml \
    --output_dir output/retrieval_flickr_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_flickr.yaml). For example, to conduct a 2x compression on 8 A100 GPUs:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_flickr.py --p 0.5 --epoch 12 \
    --pretrained pretrained/model_base_retrieval_flickr.pth \
    --config ./configs/retrieval_flickr.yaml \
    --output_dir output/retrieval_flickr_compression_2x
    ```

* Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    2x | <a href="https://drive.google.com/uc?export=download&id=1mrd7unZMFMC77Qb_3DAx7MhpZJv4Ptbw">Download</a> | [Link](./scripts/compress_retrieval_flickr_2x.sh) | <a href="https://drive.google.com/uc?export=download&id=1FnJNt5RqFPVEjCBmikKPSu1vzRsf-kN9">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1mOTbG_zvIAD3itJI1oo_0r55W3PeRy5b">Download</a> | [Link](./scripts/evaluate_retrieval_flickr_2x_compressed.sh)
    4x | <a href="https://drive.google.com/uc?export=download&id=1mrd7unZMFMC77Qb_3DAx7MhpZJv4Ptbw">Download</a> | [Link](./scripts/compress_retrieval_flickr_4x.sh)| <a href="https://drive.google.com/uc?export=download&id=1DHoUwUjKyNlm-QWdIMJKCQdBcC1vQY_F">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1mSSbvS3SkR334xxdtee0p61bRfOgjgyG">Download</a> | [Link](./scripts/evaluate_retrieval_flickr_4x_compressed.sh)


### 🚀 Image-Text and Text-Image Retrieval on the COCO Dataset with CLIP

* Dataset & Annotation

    Download the [COCO](https://cocodataset.org/#home) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_coco_clip.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_coco_clip.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_clip.py --evaluate \
    --pretrained output/retrieval_coco_clip_compression_2x/clip_large_retrieval_coco_2x_compressed.pth \
    --config ./configs/retrieval_coco_clip.yaml \
    --output_dir output/retrieval_coco_clip_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_coco_clip.yaml). For example, to conduct a 2x compression on 8 A100 GPUs:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_clip.py --p 0.5 --epoch 6 \
    --pretrained pretrained/clip_large_retrieval_coco.pth \
    --config ./configs/retrieval_coco_clip.yaml \
    --output_dir output/retrieval_coco_clip_compression_2x
    ```

* Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    2x | <a href="https://drive.google.com/uc?export=download&id=10p1oPdiMUqo0MfPul5hCb_h9mCaNCh6q">Download</a> | [Link](./scripts/compress_retrieval_coco_clip_2x.sh) | <a href="https://drive.google.com/uc?export=download&id=1EACJZO2QdbcLkBr6uvZq_z8A4vXzc8yK">Download</a> | <a href="https://drive.google.com/uc?export=download&id=184sBFSArAXfxkd8ZqlF8xsZCOOsNmT0z">Download</a> | [Link](./scripts/evaluate_retrieval_coco_clip_2x_compressed.sh)
    4x | <a href="https://drive.google.com/uc?export=download&id=10p1oPdiMUqo0MfPul5hCb_h9mCaNCh6q">Download</a> | [Link](./scripts/compress_retrieval_coco_clip_4x.sh)| <a href="https://drive.google.com/uc?export=download&id=1-OA-xkLbzH39GPfrVFux3wNZ9h0GwyJX">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1C3LRQZ2IP7St813ERH7LidTcQP99xhKw">Download</a> | [Link](./scripts/evaluate_retrieval_coco_clip_4x_compressed.sh)


### 🚀 Image-Text and Text-Image Retrieval on the Flickr30K Dataset with CLIP

* Dataset & Annotation

    Download the [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_flickr_clip.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_flickr_clip.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_clip.py --evaluate \
    --pretrained output/retrieval_flickr_clip_compression_2x/clip_large_retrieval_flickr_2x_compressed.pth \
    --config ./configs/retrieval_flickr_clip.yaml \
    --output_dir output/retrieval_flickr_clip_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_flickr_clip.yaml). For example, to conduct a 2x compression on 8 A100 GPUs:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_clip.py --p 0.5 --epoch 12 \
    --pretrained pretrained/clip_large_retrieval_flickr.pth \
    --config ./configs/retrieval_flickr_clip.yaml \
    --output_dir output/retrieval_flickr_clip_compression_2x
    ```

* Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    2x | <a href="https://drive.google.com/uc?export=download&id=1-MZP6xQRnmLZr1_pqUK4TvOA8Ic7XCoI">Download</a> | [Link](./scripts/compress_retrieval_flickr_clip_2x.sh) | <a href="https://drive.google.com/uc?export=download&id=1pE48hKlW0VI37_ebxhqBm-YVqEfFccQ4">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1kZjCv4Y9Cars7U3PL9gXP7XJsqJMyLeD">Download</a> | [Link](./scripts/evaluate_retrieval_flickr_clip_2x_compressed.sh)
    4x | <a href="https://drive.google.com/uc?export=download&id=1-MZP6xQRnmLZr1_pqUK4TvOA8Ic7XCoI">Download</a> | [Link](./scripts/compress_retrieval_flickr_clip_4x.sh)| <a href="https://drive.google.com/uc?export=download&id=1pSCr8OVzPsvnL2IEIhpBxJAKTAZ1_iCD">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1YUkN-zz6iFxquJeYKcxbETFTbWM14KWK">Download</a> | [Link](./scripts/evaluate_retrieval_flickr_clip_4x_compressed.sh)


### 🚀 Image Classification on the ImageNet Dataset

* Dataset & Annotation

    Download the [ImageNet](https://www.image-net.org/) dataset, unzip it under the `datasets` folder, and accordingly modify the option `--data-path` in compression and evaluation scripts. See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the option `--resume` of the scripts. For example, to evaluate a 50% compressed model:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_deit.py --eval --dist-eval \
    --data-path datasets/vision/imagenet \
    --model deit_small_patch16_224 \
    --resume output/train_deit_small_patch16_224_60s_300r_050x/deit_small_patch16_224_050x_compressed.pth
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the option `--finetune` of the scripts. For example, to conduct a 50% compression on 8 A100 GPUs:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_deit.py \
    --data-path datasets/vision/imagenet \
    --finetune pretrained/deit_small_patch16_224-cd65a155.pth \
    --model deit_small_patch16_224 \
    --epochs-search 60 \
    --epochs 300 \
    --batch-size 512 \
    --lr-search 1e-4 \
    --lr 1e-4 \
    --warmup-epochs 0 \
    --p 0.5 \
    --interval 800 \
    --output_dir output/train_deit_small_patch16_224_60s_300r_050x
    ```

* Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    10% | <a href="https://drive.google.com/uc?export=download&id=12I4IvkihOXvt5rr_5FLv4wVq-VgWbXXm">Download</a> | [Link](./scripts/compress_classification_imagenet_010x.sh) | <a href="https://drive.google.com/uc?export=download&id=1wqJ_zRyWqc9Wymu1fMD6tPB_LDinrapM">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1nKn5ueemhjoV0NJiVYAlJgglqk-8-Ovd">Download</a> | [Link](./scripts/evaluate_classification_imagenet_010x.sh)
    20% | <a href="https://drive.google.com/uc?export=download&id=12I4IvkihOXvt5rr_5FLv4wVq-VgWbXXm">Download</a> | [Link](./scripts/compress_classification_imagenet_020x.sh) | <a href="https://drive.google.com/uc?export=download&id=1ggihhKt3RA-xjWf4l8OZo34xAOs3A0Gr">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1_kXZ-KVsk7eG9Cjyy0jgOuAGNGzogOlf">Download</a> | [Link](./scripts/evaluate_classification_imagenet_020x.sh)
    30% | <a href="https://drive.google.com/uc?export=download&id=12I4IvkihOXvt5rr_5FLv4wVq-VgWbXXm">Download</a> | [Link](./scripts/compress_classification_imagenet_030x.sh)| <a href="https://drive.google.com/uc?export=download&id=1Teq-4eSjIB32Zm7OfuotbaqU-H_sSqhS">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1isQ9TzkdTeqUXAI1QBSAb9ypreQsMr8H">Download</a> | [Link](./scripts/evaluate_classification_imagenet_030x.sh)
    40% | <a href="https://drive.google.com/uc?export=download&id=12I4IvkihOXvt5rr_5FLv4wVq-VgWbXXm">Download</a> | [Link](./scripts/compress_classification_imagenet_040x.sh)| <a href="https://drive.google.com/uc?export=download&id=1lTVN5NRZzJmYkDDASqwikUoHXrL3vRCF">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1XfKGrEuBNo0BO61C_PxqkzRPAswDeqpx">Download</a> | [Link](./scripts/evaluate_classification_imagenet_040x.sh)
    50% | <a href="https://drive.google.com/uc?export=download&id=12I4IvkihOXvt5rr_5FLv4wVq-VgWbXXm">Download</a> | [Link](./scripts/compress_classification_imagenet_050x.sh) | <a href="https://drive.google.com/uc?export=download&id=1kQpyecczHVEf62lsAi00UDirc-T-_0M_">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1edZpbtSsny3hdUpuaMut0T2eB8Dqk3mS">Download</a> | [Link](./scripts/evaluate_classification_imagenet_050x.sh)


### 🚀 Image Segmentation on the Ade20k Dataset

* Dataset & Annotation

    Download the [Ade20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset, unzip it under the `datasets` folder, and accordingly modify the option `--dataset` in compression and evaluation scripts. See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, accordingly modify the path option of the scripts, and export the folder of datasets as the environment variable `DATASET`. For example, to evaluate a 30% compressed model:
    ```bash
    export DATASET=datasets/vision

    # for single-scale testing
    python -m torch.distributed.run --nproc_per_node=4 segm/eval/miou.py \
    output/seg_small_mask_16s_64r_030x/seg_small_mask_030x_compressed.pth ade20k --singlescale

    # for multi-scale testing
    python -m torch.distributed.run --nproc_per_node=4 segm/eval/miou.py \
    output/seg_small_mask_16s_64r_030x/seg_small_mask_030x_compressed.pth ade20k --multiscale
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, accordingly modify the option `--pretrained` of the scripts, and export the folder of datasets as the environment variable `DATASET`. For example, to conduct a 30% compression on 4 A100 GPUs:
    ```bash
    export DATASET=datasets/vision

    python -m torch.distributed.run --nproc_per_node=4 segm/train.py --dataset ade20k \
    --backbone vit_small_patch16_384 --decoder mask_transformer --no-resume \
    --pretrained pretrained/seg_small_mask.pth \
    --epochs-search 16 \
    --epochs 64 \
    --batch-size 64 \
    --lr-search 4e-3 \
    -lr 4e-3  \
    --p 0.30 \
    --interval 200 \
    --log-dir output/seg_small_mask_16s_64r_030x
    ```

* Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    10% | <a href="https://drive.google.com/uc?export=download&id=1PyWdaFahWlu4d_xX_b_ZxwqTJ5q9V-Lu">Download</a> | [Link](./scripts/compress_segmentation_ade20k_010x.sh) | <a href="https://drive.google.com/uc?export=download&id=1ACvxczNAzhUkXzbOm3OwxWvaKcH-9TRN">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1PT0MrrQvq9aGAC-l8v9qDZhE6yWkrWNb">Download</a> | [Link](./scripts/evaluation_segmentation_ade20k_010x.sh)
    15% | <a href="https://drive.google.com/uc?export=download&id=1PyWdaFahWlu4d_xX_b_ZxwqTJ5q9V-Lu">Download</a> | [Link](./scripts/compress_segmentation_ade20k_015x.sh) | <a href="https://drive.google.com/uc?export=download&id=1UMYb6nxDcsLOXJH0kNeepJCEk20WRvjC">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1KYyil7I10xREp0QJxQdHJJ9lgQlNqetP">Download</a> | [Link](./scripts/evaluation_segmentation_ade20k_015x.sh)
    20% | <a href="https://drive.google.com/uc?export=download&id=1PyWdaFahWlu4d_xX_b_ZxwqTJ5q9V-Lu">Download</a> | [Link](./scripts/compress_segmentation_ade20k_020x.sh)| <a href="https://drive.google.com/uc?export=download&id=1seuTRfqpIAMoM74PHkXlm14AmHLI8oTH">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1gb0zuxpunUB0iA0Fkar1myEdHMtaEgsU">Download</a> | [Link](./scripts/evaluation_segmentation_ade20k_020x.sh)
    30% | <a href="https://drive.google.com/uc?export=download&id=1PyWdaFahWlu4d_xX_b_ZxwqTJ5q9V-Lu">Download</a> | [Link](./scripts/compress_segmentation_ade20k_030x.sh)| <a href="https://drive.google.com/uc?export=download&id=1OCiFJbIPkmVT-FqgoNfW4Ch37mRALrj2">Download</a> | <a href="https://drive.google.com/uc?export=download&id=1MzMyAw5kaVglgpLhQt-bpcJBdtDLnkt-">Download</a> | [Link](./scripts/evaluation_segmentation_ade20k_030x.sh)


### 📑 Common Issues

#### 1. Evaluation with single GPU
   
* For BLIP and CLIP models, evaluate the 2x compressed BLIP model on the NLVR2 dataset as an example:

    ```bash
    python compress_nlvr.py --evaluate \
    --pretrained output/caption_coco_compression_2x/model_base_caption_capfilt_large_coco_2x_compressed.pth \
    --config ./configs/caption_coco.yaml \
    --output_dir output/caption_coco_compression_2x
    ```
* For DeiT, evaluate the 50% compressed model on the ImageNet dataset as an example: (Note that without the option `---dist-eval`)

    ```bash
    python compress_deit.py --eval \
    --data-path datasets/vision/imagenet \
    --model deit_small_patch16_224 \
    --resume output/train_deit_small_patch16_224_60s_300r_050x/deit_small_patch16_224_050x_compressed.pth
    ```
* For Segmenter, evaluate the 30% compressed model on the ADE20k dataset as an example:
    ```bash
    export DATASET=datasets/vision
    
    # for single-scale testing
    python segm/eval/miou.py \
    output/seg_small_mask_16s_64r_030x/seg_small_mask_030x_compressed.pth ade20k --singlescale

    # for multi-scale testing
    python segm/eval/miou.py \
    output/seg_small_mask_16s_64r_030x/seg_small_mask_030x_compressed.pth ade20k --multiscale
    ```

#### 2. Compress with single GPU
   
* For BLIP and CLIP models, compress the BLIP model to half on the NLVR2 dataset as an example:

    ```bash
    python compress_nlvr.py --p 0.5 --epoch 15 \
    --pretrained pretrained/model_base_nlvr.pth \
    --config ./configs/nlvr.yaml \
    --output_dir output/nlvr_nlvr2_compression_2x
    ```

* For DeiT, conduct a 50% compression on the ImageNet dataset as an example:
  
    ```bash
    python compress_deit.py \
    --data-path datasets/vision/imagenet \
    --finetune pretrained/deit_small_patch16_224-cd65a155.pth \
    --model deit_small_patch16_224 \
    --epochs-search 60 \
    --epochs 300 \
    --batch-size 512 \
    --lr-search 1e-4 \
    --lr 1e-4 \
    --warmup-epochs 0 \
    --p 0.5 \
    --interval 800 \
    --output_dir output/train_deit_small_patch16_224_60s_300r_050x
    ```

* For Segmenter, conduct a 30% compression on the Ade20k dataset as an example:
  
    ```bash
    export DATASET=datasets/vision

    python segm/train.py --dataset ade20k \
    --backbone vit_small_patch16_384 --decoder mask_transformer --no-resume \
    --pretrained pretrained/seg_small_mask.pth \
    --epochs-search 16 \
    --epochs 64 \
    --batch-size 64 \
    --lr-search 4e-3 \
    -lr 4e-3  \
    --p 0.30 \
    --interval 200 \
    --log-dir output/seg_small_mask_16s_64r_030x
    ```

#### 3. Out of memory during the evaluation
   
* For BLIP and CLIP models, change the `batch_size_test` (or the `batch_size` for the Image Caption task) in the corresponding config file to a smaller number.
* For DeiT, modify the option `--batch-size` of the scripts to a smaller number.
* For Segmenter, the default batch size of the evaluation is `1`. For the single-scale testing, the peak of used GPU memory on a single card is less than 5G, which should be able to run on most types of GPUs. For the multi-scale testing, the peak of used GPU memory on a single card is about 13G, which may require a GPU with relatively larger memory.

#### 4. Out of memory during the compression

* For BLIP and CLIP models, change the `batch_size_train` and `batch_size_test` (or the `batch_size` for the Image Caption task) in the corresponding config file to a smaller number. Besides, the option `--amp` for compression scripts can be used to enable mixed precision. Compress the BLIP model to half on the NLVR2 dataset as an example:
   
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_nlvr.py --p 0.5 --epoch 15 --amp \
    --pretrained pretrained/model_base_nlvr.pth \
    --config ./configs/nlvr.yaml \
    --output_dir output/nlvr_nlvr2_compression_2x
    ```
    Note that using mixed precision may produce nan gradients. Since UPop take gradients as metrics to determine pruned positions, nan gradients may disrupt the determination and degrade the performance. 

* For DeiT and Segmenter, modify the option `--batch-size` of the scripts to a smaller number. Mixed precision is not supported temporarily, as it frequently causes nan gradients.


### 🌲 Expected Folder Structures

```
├── annotation
│   ├── answer_list.json
│   ├── coco_gt
│   │   ├── coco_karpathy_test_gt.json
│   │   └── coco_karpathy_val_gt.json
│   ├── ...
├── clip                                               
├── compress_caption.py       
├── compress_deit.py        
├── compress_nlvr.py                  
├── compress ...    
├── configs                                             
├── data                                        
├── datasets
│   └── vision
│       ├── coco
│       ├── flickr
│       ├── NLVR2     
│       ├── ...                                                                              
├── deit   
├── log                                     
├── models            
├── output                                    
├── pretrained
│   ├── bert-base-uncased
│   ├── clip_large_retrieval_coco.pth
│   ├── clip_large_retrieval_flickr.pth
│   ├── ...       
├── segm                                                                                   
├── transform                                                                           
└── utils.py                                
```

### 💬 Acknowledgments
This code is built upon <a href="https://github.com/salesforce/BLIP">BLIP</a>, <a href="https://github.com/openai/CLIP">CLIP</a>, <a href="https://github.com/facebookresearch/deit">DeiT</a>, <a href="https://github.com/rstrudel/segmenter">Segmenter</a>, and <a href=https://github.com/huggingface/pytorch-image-models/tree/main/timm>timm</a>. Thanks for these awesome open-source projects!


### ✨ Citation
If you find our work or this code useful, please consider citing the corresponding paper:
```bibtex
@InProceedings{pmlr-v202-shi23e,
  title = {{UP}op: Unified and Progressive Pruning for Compressing Vision-Language Transformers},
  author = {Shi, Dachuan and Tao, Chaofan and Jin, Ying and Yang, Zhendong and Yuan, Chun and Wang, Jiaqi},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {31292--31311},
  year = {2023},
  volume = {202},
  publisher = {PMLR}
}
```

