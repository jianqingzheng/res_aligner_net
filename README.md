<div align="center">
<h1> Residual Aligner-based Network (RAN) for Coarse-to-fine Discontinuous Deformable Registration </h1>

[![DOI](https://img.shields.io/badge/DOI-j.media.2023.103038-darkyellow)](https://doi.org/10.1016/j.media.2023.103038) 
[![arXiv](https://img.shields.io/badge/arXiv-2203.04290-b31b1b.svg)](https://arxiv.org/abs/2203.04290)
[![Explore RAN in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jianqingzheng/res_aligner_net/blob/main/res_aligner_net.ipynb)

</div>

Code for *Medical Image Analysis* paper [Residual Aligner-based Network (RAN): Motion-Separable Structure for Coarse-to-fine Deformable Image Registration](https://doi.org/10.1016/j.media.2023.103038)


> This repo provides an implementation of the training and inference pipeline of RAN based on TensorFlow and Keras. 

---
### Contents ###
- [0. Brief Introduction](#0-brief-intro)
- [1. Installation](#1-installation)
- [2. Usage](#2-usage)
  - [2.1. Setup (for unpaired data)](#21-setup-for-unpaired-data)
  - [2.2. Training (>1 week)](#22-training-1-week)
  - [2.3. Inference](#23-inference)
- [3. Demo](#3-demo)
- [4. Citing this work](#4-citing-this-work)

---

## 0. Brief Intro ##

![header](supp/Graphic_abstract.png)
The research in this paper focuses on solving the problem of multi-organ discontinuous deformation alignment. An innovative quantitative metric, Motion Separability, is proposed in the paper. This metric is designed to measure the ability of deep learning networks to predict organ discontinuous deformations. Based on this metric, a novel network structure skeleton, the Motion-Separable structure, is designed. In addition, we introduce a Motion disentanglement module to help the network distinguish and process complex motion patterns among different organs.

To verify the validity of this quantitative metric as well as the accuracy and efficiency of our method, a series of unsupervised alignment experiments are conducted in the paper. These experiments cover nine major organs of the abdomen and lung images. The experimental results show that the method in the paper is not only able to effectively identify and process the complex motions among the organs, but also improves the accuracy and efficiency of the alignment.

The main contributions include:
<ul style="width: auto; height: 200px; overflow: auto; padding:0.4em; margin:0em; text-align:justify; font-size:small">
  <li> Discontinuous alignment network: this is the first quantitative study targeting discontinuous deformation alignment based on a deep learning network.
  </li>
  <li> Theoretical analysis: this paper quantifies and defines the maximum range of capturable motion and the upper bound of motion separability in neural networks, providing a theoretical analysis of the upper bound of motion separability. This helps us to understand the range of motion that can be recognised by the network and guides the optimisation of the network structure and parameter settings.   
  </li>
  <li> Motion separable backbone structure: based on the theoretical analysis in this paper, a novel multi-scale skeleton structure is designed in the paper. This structure enables the network to efficiently predict motion patterns with larger separable upper bounds by using optimized dilated convolution on high-resolution feature maps, while maintaining a capturable motion range with low computational complexity.
  </li>
  <li> Motion decoupling and refinement module: in addition, we propose a Residual Aligner module (RAM) that utilizes confidence levels and mechanisms based on semantic and contextual information to differentiate predicted displacements in different organs or regions. This means that our method can more accurately deal with specific movements in each region.
  </li>
  <li> Accurate and Efficient Registration Results: The above-proposed components constitute a novel residual alignment network (RAN) that performs efficient, coarse-to-fine, unsupervised alignment of separable motions on publicly available lung and abdominal CT data, achieving higher accuracy and lower computational cost.
  </li>
</ul>

---
## 1. Installation ##

Clone code from Github repo: https://github.com/jianqingzheng/res_aligner_net.git
```shell
git clone https://github.com/jianqingzheng/res_aligner_net.git
cd res_aligner_net/
```

install packages

[![OS](https://img.shields.io/badge/OS-Windows%7CLinux-darkblue)]()
[![PyPI pyversions](https://img.shields.io/badge/Python-3.8-blue)](https://pypi.python.org/pypi/ansicolortags/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.3.1-lightblue)](www.tensorflow.org)
[![Numpy](https://img.shields.io/badge/Numpy-1.19.5-lightblue)](https://numpy.org)

```shell
pip install tensorflow==2.3.1
pip install numpy==1.19.5
pip install pyquaternion==0.9.9
```

> Other versions of the packages could also be applicable

---
## 2. Usage ##

### 2.1. Setup (for unpaired data) ###
```
[$DOWNLOAD_DIR]/res_aligner_net/           
├── data/[$data_name]/dataset
|   |   # experimental dataset for training and testing (.nii|.nii.gz files)
|   ├── train/
|   |	├── images/
|   |   |   ├──0001.nii.gz
|   |   |   └── ...
|   |	├── labels/
|   |   |   ├──0001.nii.gz
|   |   |   └── ...
|   ├── test/
|   |	├── images/
|   |   |   ├──0001.nii.gz
|   |   |   └── ...
|   |	└── labels/
|   |       ├──0001.nii.gz
|   |       └── ...
└── models/[$data_name]/
|   └── [$data_name]-[$model_name]/
|       |   # the files of model parameters (.tf.index and .tf.data-000000-of-00001 files)
|       ├── model_1_[$model_num].tf.index
|       ├── model_1_[$model_num].tf.data-000000-of-00001
|       └── ...
└── ...
```

1. Run ```python external/deepreg/abd_data.py``` to download and setup abdominal CT, <br /> or Run ```python external/deepreg/lung_data.py``` to download and setup lung CT
2. Run ```python main_preprocess.py --proc_type train --data_name $data_name```
3. Run ```python main_preprocess.py --proc_type test --data_name $data_name```

<div align="center">
	
| Argument              | Description                                	|
| --------------------- | ----------------------------------------------|
| `--data_name` 	      | The data folder name                    |

</div>

\* Example for the setup (unpaired_ct_abdomen):

1. Run 
```shell
python external/deepreg/abd_data.py
```
2. Run 
```shell
python main_preprocess.py --proc_type train --data_name unpaired_ct_abdomen
python main_preprocess.py --proc_type test --data_name unpaired_ct_abdomen
```


> The data used for experiments in this paper are publicly available from [abdomen CT](https://github.com/ucl-candi/datasets_deepreg_demo/archive/abdct.zip) and [lung CT](https://zenodo.org/record/3835682).


### 2.2. Training (>1 week) ###
1. Run ```python main_train.py --model_name $model_name --data_name $data_name --max_epochs $max_epochs```
2. Check the saved model in ```res_aligner_net/models/$data_name/$data_name-$model_name/```

<div align="center">
	
| Argument              | Description                                	|
| --------------------- | ----------------------------------------------|
| `--data_name` 	      | The data folder name                    |
| `--model_name`        | The used model                      	     	|
| `--max_epochs`        | The max epoch number for training 	     	|

</div>

> `max_epochs==0` for training from scratch

\* Example for training (default):

1. Run
```shell
python main_train.py --model_name RAN4 --data_name unpaired_ct_abdomen --max_epochs 0
```
2. Check the saved model in ```res_aligner_net/models/unpaired_ct_abdomen/unpaired_ct_abdomen-RAN4/```



### 2.3. Inference ###
1. Run ```python main_infer.py --model_name $model_name --data_name $data_name```
2. Check the results in ```res_aligner_net/data/$data_name/dataset/test_proc/warped_img```

<div align="center">

| Argument              | Description                                	|
| --------------------- | ----------------------------------------------|
| `--data_name` 	| The data folder name                       	|
| `--model_name`        | The used network structure                    |
| `--model_id`         | The index of the model                      	|

</div>

> `model_id==1` for a model after synthetic training,
> `model_id==2` for a model after real training,
> `model_id==3` for the model trained according to the paper's settings.

\* Example for inference (default):

1. Run
```shell
python main_infer.py --model_name RAN4 --data_name unpaired_ct_abdomen
```
2. Check the results in ```res_aligner_net/data/unpaired_ct_abdomen/dataset/test_proc/warped_img```

---
## 3. Demo ##
A demo can be found in the provided [notebook](https://github.com/jianqingzheng/res_aligner_net/blob/main/res_aligner_net.ipynb).

Alternatively, it can be easily run via [![Explore RAN in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jianqingzheng/res_aligner_net/blob/main/res_aligner_net.ipynb).

---

## 4. Citing this work

Any publication that discloses findings arising from using this source code or the network model should cite:
- Zheng, J. Q., Wang, Z., Huang, B., Lim, N. H., & Papież, B. W. "Residual Aligner-based Network (RAN): Motion-separable structure for coarse-to-fine discontinuous deformable registration." *Medical Image Analysis*, 2024, 91: 103038.
```bibtex
@article{ZHENG2024103038,
	title = {Residual Aligner-based Network (RAN): Motion-separable structure for coarse-to-fine discontinuous deformable registration},
	journal = {Medical Image Analysis},
	volume = {91},
	pages = {103038},
	year = {2024},
	issn = {1361-8415},
	doi = {https://doi.org/10.1016/j.media.2023.103038},
	url = {https://www.sciencedirect.com/science/article/pii/S1361841523002980},
	author = {Jian-Qing Zheng and Ziyang Wang and Baoru Huang and Ngee Han Lim and Bartłomiej W. Papież},
	keywords = {Discontinuous deformable registration, Motion-separable structure, Motion disentanglement, Coarse-to-fine registration},
}
```
