# NS-SV: Neural Scoring for speaker verification

This is an open-source project that develops a novel framework for speaker verification. The code is developed based on [sunine](https://gitlab.com/csltstu/sunine).
In this project, our goal is to design a "general solver", which could operate effectively across a wide range of real-world scenarios (eg. clean, noisy, multi-speaker).

If you find this project beneficial for your research, we kindly encourage you to cite [our paper](https://arxiv.org/abs/2410.16428).


## Overview

<img src="source/structure.png" alt="structure" width="1000"/>


## Quick installation
1. Clone this repo

```base
git clone https://github.com/DozingOFFF/NS-SV.git
```

2. Create conda env and install the requirements

```base
conda create -n ns python=3.8
conda activate ns
conda install pytorch==2.2.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Methodologies

### Data simulation

Data simulation tool is provided. Users can generate datasets and trial lists for the test scenarios, thereby facilitating the reproduction of the validation results from our paper.

```base
python steps/create_evaldata.py
```

### Model Training

#### Step1: prepare enroll model

Users can directly use our pre-trained model `source\resnet34_avg.ckpt`, or retrain a new model using [sunine](https://gitlab.com/csltstu/sunine).

#### Step2: prepare hard speaker dict

Prepare hard speaker dict to do difficult sample mining while training model for clean scenario.

```base
python steps/create_hardspk_dict.py
```

#### Step3: train a clean-scenario model

```base
python main.py --config conf/ns_clean.yaml
```

#### Step4: train a multi-scenario model

```base
python main.py --config conf/ns_multi.yaml
```

#### Step5: average model checkpoints

```base
python steps/avg_model.py
```

## Citation

```base
@article{lin2024neural,
  title={Neural Scoring, Not Embedding: A Novel Framework for Robust Speaker Verification},
  author={Lin, Wan and Chen, Junhui and Wang, Tianhao and Zhou, Zhenyu and Li, Lantian and Wang, Dong},
  journal={arXiv preprint arXiv:2410.16428},
  year={2024}
}
```