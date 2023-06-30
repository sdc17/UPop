# UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers

<p align="center"> <a href="https://arxiv.org/pdf/2301.13741.pdf" target="_blank">[Paper]</a> 
<a href="https://arxiv.org/abs/2301.13741" target="_blank">[ArXiv]</a> 
<a href="https://github.com/sdc17/UPop" target="_blank">[Code]</a>
<a href="https://dachuanshi.com/UPop-Project/" target="_blank">[Website]</a>
</p>

<img src="UPop.png" width="800">

Official implementation of [UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers](https://arxiv.org/abs/2301.13741). UPop is the **first structured pruning framework for vision-language Transformers**. It enables **effective structured pruning on various multi-modal & uni-modal tasks, datasets, and model architectures**.

### What's New 🥳
* (Jun 2023), we worked on a new project CrossGET: Cross-Guided Ensemble of Tokens for Accelerating Vision-Language Transformers. CrossGET reduces token computational costs effectively for accelerating. [[Paper]](https://arxiv.org/pdf/2305.17455.pdf). [[Code(coming soon)]](https://github.com/sdc17/CrossGET).  💡

* (Jun 30, 2023), we released the ```implementation```, ```scripts```, ```checkpoints```, and ```logs``` of UPop. [[Code]](https://github.com/sdc17/UPop). [[Website]](https://dachuanshi.com/UPop-Project/). 🚩

* (Apr 25, 2023), our work UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers was accepted by ICML 2023. [[Paper]](https://arxiv.org/pdf/2301.13741.pdf). [[ArXiv]](https://arxiv.org/abs/2301.13741). 🎉


### Installation
The code is tested on `Pytorch==1.11.0`, `cuda==11.3.1`, and `python==3.8.13`. The dependencies can be installed by: (a possible issue: [cannot find package 'petrel-oss-sdk'](https://github.com/sdc17/UPop#5-cannot-find-package-petrel-oss-sdk-while-installing-dependencies)) <pre/> conda install --yes --file requirements.txt </pre> 

### Supported Tasks, Models, and Datasets
Type |  Supported Tasks | Supported Models  | Supported Datasets |
--- | --- | :---: | :---: 
Multi-modal | [Visual Reasoning](https://github.com/sdc17/UPop#visual-reasoning-on-the-nlvr2-dataset) | [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/sdc17/UPop#visual-reasoning-on-the-nlvr2-dataset)) | [NLVR2](https://lil.nlp.cornell.edu/nlvr/)
Multi-modal |[Image Caption](https://github.com/sdc17/UPop#image-caption-on-the-coco-caption-dataset) | [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/sdc17/UPop#image-caption-on-the-coco-caption-dataset)) | [COCO Caption](https://cocodataset.org/#home)
Multi-modal |[Visual Question Answer](https://github.com/sdc17/UPop#visual-question-answer-on-the-vqav2-dataset) | [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/sdc17/UPop#visual-question-answer-on-the-vqav2-dataset)) | [VQAv2](https://visualqa.org/)
Multi-modal |[Image-Text Retrieval](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-coco-dataset) | [CLIP](https://github.com/openai/CLIP) ([instructions](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-coco-dataset-with-clip)), [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-coco-dataset)) | [COCO](https://cocodataset.org/#home), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)
Multi-modal |[Text-Image Retrieval](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-coco-dataset) | [CLIP](https://github.com/openai/CLIP) ([instructions](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-flickr30k-dataset-with-clip)), [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/sdc17/UPop#image-text-and-text-image-retrieval-on-the-flickr30k-dataset)) | [COCO](https://cocodataset.org/#home), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)
Uni-modal |[Image Classification](https://github.com/sdc17/UPop#image-classification-on-the-imagenet-dataset) | [DeiT](https://github.com/facebookresearch/deit) ([instructions](https://github.com/sdc17/UPop#image-classification-on-the-imagenet-dataset)) | [ImageNet](https://www.image-net.org/)
Uni-modal |[Image Segmentation](https://github.com/sdc17/UPop#image-segmentation-on-the-ade20k-dataset) | [Segmenter](https://github.com/rstrudel/segmenter) ([instructions](https://github.com/sdc17/UPop#image-segmentation-on-the-ade20k-dataset)) | [Ade20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

### Visual Reasoning on the NLVR2 Dataset

* Dataset & Annotation

    Download the [NLVR2](https://lil.nlp.cornell.edu/nlvr/) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/nlvr.yaml). Download all-in-one annotations (including annotations for Visual Reasoning, Image Caption, VQA, Image-Text Retrieval, and Text-Image Retrieval tasks) from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/nlvr.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model: (possible issues: [on one GPU](https://github.com/sdc17/UPop#1-evaluation-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#3-out-of-memory-during-the-evaluation))
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_nlvr.py --evaluate \
    --pretrained output/nlvr_nlvr2_compression_2x/model_base_nlvr_nlvr2_2x_compressed.pth \
    --config ./configs/nlvr.yaml \
    --output_dir output/nlvr_nlvr2_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/nlvr.yaml). For example, to conduct a 2x compression on 8 A100 GPUs (80G): (possible issues: [on one GPU](https://github.com/sdc17/UPop#2-compress-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#4-out-of-memory-during-the-compression))
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



### Image Caption on the COCO Caption Dataset

* Dataset & Annotation

    Download the [COCO Caption](https://cocodataset.org/#home) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/caption_coco.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/caption_coco.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model: (possible issues: [on one GPU](https://github.com/sdc17/UPop#1-evaluation-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#3-out-of-memory-during-the-evaluation), ['java' runtime error](https://github.com/sdc17/UPop#6-no-such-file-or-directory-java-java-while-evaluating-or-compressing-models-on-the-image-caption-task))
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_caption.py --evaluate \
    --pretrained output/caption_coco_compression_2x/model_base_caption_capfilt_large_coco_2x_compressed.pth \
    --config ./configs/caption_coco.yaml \
    --output_dir output/caption_coco_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/caption_coco.yaml). For example, to conduct a 2x compression on 8 A100 GPUs (80G): (possible issues: [on one GPU](https://github.com/sdc17/UPop#2-compress-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#4-out-of-memory-during-the-compression), ['java' runtime error](https://github.com/sdc17/UPop#6-no-such-file-or-directory-java-java-while-evaluating-or-compressing-models-on-the-image-caption-task))
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
    


### Visual Question Answer on the VQAv2 Dataset

* Dataset & Annotation

    Download the [VQAv2](https://visualqa.org/) dataset and [Visual Genome](https://visualgenome.org/) dataset, unzip them under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/vqa.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/vqa.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model: (possible issues: [on one GPU](https://github.com/sdc17/UPop#1-evaluation-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#3-out-of-memory-during-the-evaluation). Note that the scripts will generate answers `vqa_result.json`, which should be submitted to the [official server](https://eval.ai/web/challenges/challenge-page/830/overview) to obtain evaluation results.) 
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_vqa.py --evaluate \
    --pretrained output/vqa_vqa2_compression_2x/model_base_vqa_capfilt_large_vqa2_2x_compressed.pth \
    --config ./configs/vqa.yaml \
    --output_dir output/vqa_vqa2_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/vqa.yaml). For example, to conduct a 2x compression on 8 A100 GPUs (80G): (possible issues: [on one GPU](https://github.com/sdc17/UPop#2-compress-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#4-out-of-memory-during-the-compression))
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
    

### Image-Text and Text-Image Retrieval on the COCO Dataset

* Dataset & Annotation

    Download the [COCO](https://cocodataset.org/#home) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_coco.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_coco.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model: (possible issues: [on one GPU](https://github.com/sdc17/UPop#1-evaluation-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#3-out-of-memory-during-the-evaluation))
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval.py --evaluate \
    --pretrained output/retrieval_coco_compression_2x/model_base_retrieval_coco_2x_compressed.pth --config ./configs/retrieval_coco.yaml \
    --output_dir output/retrieval_coco_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_coco.yaml). For example, to conduct a 2x compression on 8 A100 GPUs (80G): (possible issues: [on one GPU](https://github.com/sdc17/UPop#2-compress-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#4-out-of-memory-during-the-compression))
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
    

### Image-Text and Text-Image Retrieval on the Flickr30K Dataset

* Dataset & Annotation

    Download the [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_flickr.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_flickr.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model: (possible issues: [on one GPU](https://github.com/sdc17/UPop#1-evaluation-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#3-out-of-memory-during-the-evaluation))
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_flickr.py --evaluate \
    --pretrained output/retrieval_flickr_compression_2x/model_base_retrieval_flickr_2x_compressed.pth \
    --config ./configs/retrieval_flickr.yaml \
    --output_dir output/retrieval_flickr_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_flickr.yaml). For example, to conduct a 2x compression on 8 A100 GPUs (80G): (possible issues: [on one GPU](https://github.com/sdc17/UPop#2-compress-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#4-out-of-memory-during-the-compression))
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


### Image-Text and Text-Image Retrieval on the COCO Dataset with CLIP

* Dataset & Annotation

    Download the [COCO](https://cocodataset.org/#home) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_coco_clip.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_coco_clip.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model: (possible issues: [on one GPU](https://github.com/sdc17/UPop#1-evaluation-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#3-out-of-memory-during-the-evaluation), ['clip/mock.py' runtime error](https://github.com/sdc17/UPop#7-runtime-error-caused-by-clipmockpy-while-evaluating-or-compressing-models-with-clip-based-models))
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_clip.py --evaluate \
    --pretrained output/retrieval_coco_clip_compression_2x/clip_large_retrieval_coco_2x_compressed.pth \
    --config ./configs/retrieval_coco_clip.yaml \
    --output_dir output/retrieval_coco_clip_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_coco_clip.yaml). For example, to conduct a 2x compression on 8 A100 GPUs (80G): (possible issues: [on one GPU](https://github.com/sdc17/UPop#2-compress-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#4-out-of-memory-during-the-compression), ['clip/mock.py' runtime error](https://github.com/sdc17/UPop#7-runtime-error-caused-by-clipmockpy-while-evaluating-or-compressing-models-with-clip-based-models))
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


### Image-Text and Text-Image Retrieval on the Flickr30K Dataset with CLIP

* Dataset & Annotation

    Download the [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_flickr_clip.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_flickr_clip.yaml). See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a 2x compressed model: (possible issues: [on one GPU](https://github.com/sdc17/UPop#1-evaluation-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#3-out-of-memory-during-the-evaluation), ['clip/mock.py' runtime error](https://github.com/sdc17/UPop#7-runtime-error-caused-by-clipmockpy-while-evaluating-or-compressing-models-with-clip-based-models))
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_clip.py --evaluate \
    --pretrained output/retrieval_flickr_clip_compression_2x/clip_large_retrieval_flickr_2x_compressed.pth \
    --config ./configs/retrieval_flickr_clip.yaml \
    --output_dir output/retrieval_flickr_clip_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_flickr_clip.yaml). For example, to conduct a 2x compression on 8 A100 GPUs (80G): (possible issues: [on one GPU](https://github.com/sdc17/UPop#2-compress-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#4-out-of-memory-during-the-compression), ['clip/mock.py' runtime error](https://github.com/sdc17/UPop#7-runtime-error-caused-by-clipmockpy-while-evaluating-or-compressing-models-with-clip-based-models))
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


### Image Classification on the ImageNet Dataset

* Dataset & Annotation

    Download the [ImageNet](https://www.image-net.org/) dataset, unzip it under the `datasets` folder, and accordingly modify the option `--data-path` in compression and evaluation scripts. See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the option `--resume` of the scripts. For example, to evaluate a 50% compressed model: (possible issues: [on one GPU](https://github.com/sdc17/UPop#1-evaluation-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#3-out-of-memory-during-the-evaluation))
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_deit.py --eval --dist-eval \
    --data-path datasets/vision/imagenet \
    --model deit_small_patch16_224 \
    --resume output/train_deit_small_patch16_224_60s_300r_050x/deit_small_patch16_224_050x_compressed.pth
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the option `--finetune` of the scripts. For example, to conduct a 50% compression on 8 A100 GPUs (80G): (possible issues: [on one GPU](https://github.com/sdc17/UPop#2-compress-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#4-out-of-memory-during-the-compression))
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


### Image Segmentation on the Ade20k Dataset

* Dataset & Annotation

    Download the [Ade20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset, unzip it under the `datasets` folder, and accordingly modify the option `--dataset` in compression and evaluation scripts. See [here](https://github.com/sdc17/UPop#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, accordingly modify the path option of the scripts, and export the folder of datasets as the environment variable `DATASET`. For example, to evaluate a 30% compressed model: (possible issues: [on one GPU](https://github.com/sdc17/UPop#1-evaluation-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#3-out-of-memory-during-the-evaluation))
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
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, accordingly modify the option `--pretrained` of the scripts, and export the folder of datasets as the environment variable `DATASET`. For example, to conduct a 30% compression on 4 A100 GPUs (80G): (possible issues: [on one GPU](https://github.com/sdc17/UPop#2-compress-with-single-gpu), [out of memory](https://github.com/sdc17/UPop#4-out-of-memory-during-the-compression))
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


### Common Issues

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

#### 5. Cannot find package `petrel-oss-sdk` while installing dependencies

Just skip it. `petrel-oss-sdk` is an internal package for us to accelerate data loading. And the code will also work without this package.

#### 6. No such file or directory: `'java'`: `'java'` while evaluating or compressing models on the Image Caption task. 
   
Please refer to [this solution](https://github.com/helloMickey/caption_eval#fixed-bugs-in--ruotianluococo-caption).

#### 7. Runtime error caused by [clip/mock.py](./clip/mock.py) or [deit/mock.py](./deit/mock.py) while evaluating or compressing.
   
* For CLIP models, the [clip/mock.py](./clip/mock.py) is used for patching our modification to the `nn.MultiheadAttention`. It was modified from the source code of the `nn.MultiheadAttention` in version `Pytorch==1.11.0`, and also tested on `Pytorch==1.12.1` and `Pytorch==1.13.1`. However, it may not be compatible with other `Pytorch` versions that we have not tested. If you encounter this error in other versions, you may switch to version `1.11.0` or create your own patch file by referring to our [clip/mock.py](./clip/mock.py).

* For DeiT models, the [deit/mock.py](./deit/mock.py) is used for patching our modification to the `timm.models`. It was modified from the source code of the `timm.models.vision_transformer` in version `timm==0.4.12` and `torchvision==0.12.0`. It may not be compatible with other `timm` and `torchvision` versions that we have not tested. If you encounter this error in other versions, you may switch to the above versions we used, or create your own patch file by referring to our [deit/mock.py](./deit/mock.py).

#### 8. Other issues

You can post them on the [Issues](https://github.com/sdc17/UPop/issues) page.


### Expected Folder Structures

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

### Acknowledgments
This code is built upon <a href="https://github.com/salesforce/BLIP">BLIP</a>, <a href="https://github.com/openai/CLIP">CLIP</a>, <a href="https://github.com/facebookresearch/deit">DeiT</a>, <a href="https://github.com/rstrudel/segmenter">Segmenter</a>, and <a href=https://github.com/huggingface/pytorch-image-models/tree/main/timm>timm</a>. We thank the original authors for their open-source work.


### Citation
If you find this work useful, please consider citing the corresponding paper:
```bibtex
@article{shi2023upop,
  title={Upop: Unified and progressive pruning for compressing vision-language transformers},
  author={Shi, Dachuan and Tao, Chaofan and Jin, Ying and Yang, Zhendong and Yuan, Chun and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2301.13741},
  year={2023}
}
```

