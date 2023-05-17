# 4d-srda_sr-mixup <!-- omit in toc -->

[![license](https://img.shields.io/badge/license-CC%20BY--NC--SA-informational)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) [![reference](https://img.shields.io/badge/reference-arXiv-important)](https://arxiv.org/abs/2212.03656)  [![pytorch](https://img.shields.io/badge/PyTorch-1.11.0-informational)](https://pytorch.org/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7608394.svg)](https://doi.org/10.5281/zenodo.7608394)

This repository contains the source code used in *Spatio-Temporal Super-Resolution Data Assimilation (SRDA) Utilizing Deep Neural Networks with Domain Generalization Technique Toward Four-Dimensional SRDA* ([arXiv](https://arxiv.org/abs/2212.03656)).

- [Setup](#setup)
  - [Docker Containers](#docker-containers)
  - [Singularity Containers](#singularity-containers)
- [How to Perform Experiments](#how-to-perform-experiments)
  - [CFD Simulations](#cfd-simulations)
  - [Data Preparation](#data-preparation)
  - [Training and Tuning](#training-and-tuning)
  - [Evaluation](#evaluation)
- [Cite](#cite)

## Setup

- Basically, the Singularity containers were used for experiments.
- At least, 1 GPU board is required.
- Notes
  - The Docker containers have the same environments as in the Singularity containers.
  - `tsubame` means the super-computer at the Tokyo Institute of Technology ([webpage](https://www.t3.gsic.titech.ac.jp/en)).
    - Singularity containers can be used on TSUBAME.

### Docker Containers

1. Install [Docker](https://docs.docker.com/get-started/).
1. Build docker containers: `$ docker compose build`
1. Start docker containers: `$ docker compose up -d`

### Singularity Containers

1. Install [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/quick_start.html).
1. Build Singularity containers:
    - `$ singularity build -f pytorch_local.sif ./singularity/pytorch/pytorch.def`
    - `$ singularity build -f pytorch_tsubame.sif ./singularity/pytorch_tsubame/pytorch_tsubame.def`
    - `pytorch_tsubame.sif` is for the super-computer, TSUBAME, at Tokyo Institute of Technology ([webpage](https://www.t3.gsic.titech.ac.jp/en))
1. Start singularity containers:
    - The following command is for local environments

```sh
$ singularity exec --nv --env PYTHONPATH="$(pwd)/pytorch" \
    pytorch_local.sif jupyter lab \
    --no-browser --ip=0.0.0.0 --allow-root --LabApp.token='' --port=8888
```

## How to Perform Experiments

- The Singularity container (on a local environment), `pytorch_local.sif`, is used in the following experiments.
  - We confirmed the following code works on an [NVIDIA A100 40GB PCIe](https://www.nvidia.com/en-us/data-center/a100/).
- Note
  - On [TSUBAME](https://www.t3.gsic.titech.ac.jp/en), the same code was run using `pytorch_tsubame.sif`.
  - In deep learning, distributed data parallel (DDP) was used ([`train_ddp_ml_model.py`](./pytorch/script/python/train_ddp_ml_model.py)) on TSUBAME, where four GPUs of TESLA P100 were used.

### CFD Simulations

- In each script, a number of simulations must be specified.
- Run the following scripts:
  - [`./pytorch/script/shell/simulate_cfd_jet_for_making_analysis_train_data.sh`](./pytorch/script/shell/simulate_cfd_jet_for_making_analysis_train_data.sh)
  - [`./pytorch/script/shell/simulate_cfd_jet_for_making_forecast_train_data.sh`](./pytorch/script/shell/simulate_cfd_jet_for_making_forecast_train_data.sh)
  - [`./pytorch/script/shell/simulate_cfd_jet_uhr_jet.sh`](./pytorch/script/shell/simulate_cfd_jet_uhr_jet.sh)

### Data Preparation

- Run the following notebook on JupyterLab: [`split_npy.ipynb`](./pytorch/notebook/split_npy.ipynb)

### Training and Tuning

- In each script, a configuration must be specified.
- ST-SRDA
  - [`./pytorch/script/shell/train_ddp_ml_model.sh`](./pytorch/script/shell/train_ddp_ml_model.sh) (for multiple GPUs)
  - [`./pytorch/script/shell/train_ml_model.sh`](./pytorch/script/shell/train_ml_model.sh) (for a single GPU)
- EnKF-SR
  - [`./pytorch/script/shell/tune_enkf_sr.sh`](./pytorch/script/shell/tune_enkf_sr.sh)
- EnKF-HR
  - [generate_additive_inflation_cov.ipynb](./pytorch/notebook/generate_additive_inflation_cov.ipynb)
  - [`./pytorch/script/shell/tune_enkf_hr.sh`](./pytorch/script/shell/tune_enkf_hr.sh)

### Evaluation

- Run the notebooks in the following order.
  - [`./pytorch/notebook/test_srda_using_uhr.ipynb`](./pytorch/notebook/test_srda_using_uhr.ipynb)
  - [`./pytorch/notebook/test_enkf_sr_using_uhr.ipynb`](./pytorch/notebook/test_enkf_sr_using_uhr.ipynb)
  - [`./pytorch/notebook/test_enkf_hr_using_uhr.ipynb`](./pytorch/notebook/test_enkf_hr_using_uhr.ipynb)

## Cite

```bibtex
@misc{https://doi.org/10.48550/arxiv.2212.03656,
  doi = {10.48550/ARXIV.2212.03656},
  url = {https://arxiv.org/abs/2212.03656},
  author = {Yasuda, Yuki and Onishi, Ryo},  
  title = {Spatio-Temporal Super-Resolution Data Assimilation (SRDA) Utilizing Deep Neural Networks with Domain Generalization Technique Toward Four-Dimensional SRDA},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
