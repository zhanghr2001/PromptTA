# PromptTA: Prompt-driven Text Adapter for Source-free Domain Generalization [ICASSP 2025]


<!-- [arXiv](https://arxiv.org/abs/2312.09553v2) -->


Authors: Haoran Zhang, [Shuanghao Bai](https://baishuanghao.github.io/), [Wanqi Zhou](https://scholar.google.com/citations?user=3Q_3PR8AAAAJ&hl=zh-CN), [Jingwen Fu](https://scholar.google.com/citations?hl=zh-CN&user=2Cu9uMUAAAAJ), [Badong Chen](https://scholar.google.com/citations?user=mq6tPX4AAAAJ&hl=zh-CN&oi=ao).

## Highlights

![main figure](model.jpg)
> **<p align="justify"> Abstract:** *Source-free domain generalization (SFDG) tackles the challenge of adapting models to unseen target domains without access to source domain data. 
To deal with this challenging task, recent advances in SFDG have primarily focused on leveraging the text modality of vision-language models such as CLIP. 
These methods involve developing a transferable linear classifier based on diverse style features extracted from the text and learned prompts or deriving domain-unified text representations from domain banks. 
However, both style features and domain banks have limitations in capturing comprehensive domain knowledge.
In this work, we propose Prompt-Driven Text Adapter (PromptTA) method, which is designed to better capture the distribution of style features and employ resampling to ensure thorough coverage of domain knowledge. 
To further leverage this rich domain information, we introduce a text adapter that learns from these style features for efficient domain information storage.
Extensive experiments conducted on four benchmark datasets demonstrate that PromptTA achieves state-of-the-art performance.* </p>

<details>
  
<summary>Main Contributions</summary>

1) We propose PromptTA, a novel adapter-based framework for SFDG that incorporates a text adapter to effectively leverage rich domain information.
2) We introduce style feature resampling that ensures comprehensive coverage of textual domain knowledge.
3) Extensive experiments demonstrate that our PromptTA achieves the state of the art on DG benchmarks.
   
</details>


<!-- ## Results
### PDA in comparison with existing prompt tuning methods
Results reported below show accuracy across 3 UDA datasets with ViT-B/16 backbone. Our PDA method adopts the paradigm of multi-modal prompt tuning.

| Method                                                    | Office-Home Acc. | Office-31 Acc. |  VisDA-2017 Acc.  | 
|-----------------------------------------------------------|:---------:|:----------:|:---------:|
| [CLIP](https://arxiv.org/abs/2103.00020)                  |   82.1   |   77.5    |   88.9   | 
| [CoOp](https://arxiv.org/abs/2109.01134)                  |   83.9   |   89.4    |   82.7   |
| [CoCoOp](https://arxiv.org/abs/2203.05557)                |   84.1   |   88.9    |   84.2   | 
| [VP](https://arxiv.org/abs/2203.17274)                    |   81.7   |   77.4    |   88.7   | 
| [VPT-deep](https://arxiv.org/abs/2203.12119)              |   83.9   |   89.4    |   86.2   | 
| [MaPLe](https://arxiv.org/abs/2210.03117)                 |   84.2   |   89.6    |   83.5   |
| [DAPL](https://arxiv.org/abs/2202.06687)                  |   84.4   |   81.2    |   89.5   |
| [PDA](https://arxiv.org/abs/2312.09553) (Ours)            |   **85.7**   |   **91.2**    | **89.7** |  -->

## Installation 
For installation and other package requirements, please follow the instructions as follows. 
This codebase is tested on Ubuntu 18.04 LTS with python 3.7. Follow the below steps to create environment and install dependencies.

* Setup conda environment.
```bash
# Create a conda environment
conda create -y -n prompt_ta python=3.7

# Activate the environment
conda activate prompt_ta

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone PromptTA code repository and install requirements.
```bash
# Clone PromptTA code base
git clone https://github.com/zhanghr2001/PromptTA.git
cd PromptTA

# Install requirements
pip install -r requirements.txt
```

<!-- ## Data Preparation
Please follow the instructions to prepare all datasets.
Datasets list:
- [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?pli=1&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
- [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code)
- [VisDA-2017](http://ai.bu.edu/visda-2017/#download) -->


## Training and Evaluation
Please follow the instructions for training, evaluating, and reproducing the results.
Firstly, you need to **modify the directory of datasets by yourself**.

```bash
# Example: train and evaluate on PACS dataset, with backbone ResNet-50 and GPU 0
bash scripts/prompt_ta/main_ta_all.sh pacs b128_ep50_pacs RN50 0
```

The details are in each method folder in [scripts folder](scripts/).  

<!-- 
## Supported Methods
Supported methods in this codespace are as follows:

| Method                    |                   Paper                        |                             Code                                     |               Script                           |
|---------------------------|:----------------------------------------------:|:--------------------------------------------------------------------:|:----------------------------------------------:|
| CoOp                      | [IJCV 2022](https://arxiv.org/abs/2109.01134)  |  [link](https://github.com/KaiyangZhou/CoOp)                         |  [link](scripts/coop)                          |
| CoCoOp                    | [CVPR 2022](https://arxiv.org/abs/2203.05557)  |  [link](https://github.com/KaiyangZhou/CoOp)                         |  [link](scripts/cocoop)                        |
| VP                        | [-](https://arxiv.org/abs/2203.17274)          |  [link](https://github.com/hjbahng/visual_prompting)                 |  -                                             |
| VPT                       | [ECCV 2022](https://arxiv.org/abs/2203.12119)  |  [link](https://github.com/KMnP/vpt)                                 |  [link](scripts/vpt)                           |
| IVLP & MaPLe              | [CVPR 2023](https://arxiv.org/abs/2210.03117)  |  [link](https://github.com/muzairkhattak/multimodal-prompt-learning) |  [link](scripts/ivlp) & [link](scripts/maple)  |
| DAPL                      | [TNNLS 2023](https://arxiv.org/abs/2202.06687) |  [link](https://github.com/LeapLabTHU/DAPrompt)                      |  [link](scripts/dapl)                          |
 -->

<!-- ## Citation
If our code is helpful to your research or projects, please consider citing:
```bibtex
@inproceedings{bai2024prompt,
  title={Prompt-based distribution alignment for unsupervised domain adaptation},
  author={Bai, Shuanghao and Zhang, Min and Zhou, Wanqi and Huang, Siteng and Luan, Zhirong and Wang, Donglin and Chen, Badong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={2},
  pages={729--737},
  year={2024}
}
``` -->


## Acknowledgements

Our style of reademe refers to [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning). 
And our code is based on [CoOp and CoCoOp](https://github.com/KaiyangZhou/CoOp), [DAPL](https://github.com/LeapLabTHU/DAPrompt/tree/main) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) etc. repository. We thank the authors for releasing their codes. If you use their codes, please consider citing these works as well.