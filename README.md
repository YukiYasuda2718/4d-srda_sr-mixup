# 4d-srda_sr-mixup <!-- omit in toc -->

[![license](https://img.shields.io/badge/license-CC%20BY--NC--SA-informational)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) [![reference](https://img.shields.io/badge/reference-arXiv-important)](https://arxiv.org/abs/2212.03656)  [![pytorch](https://img.shields.io/badge/PyTorch-1.11.0-informational)](https://pytorch.org/)

This repository contains the source code used in *Spatio-Temporal Super-Resolution Data Assimilation (SRDA) Utilizing Deep Neural Networks with Domain Generalization Technique Toward Four-Dimensional SRDA* ([arXiv](https://arxiv.org/abs/2212.03656)).

- [Setup](#setup)
  - [Docker Containers](#docker-containers)
  - [Singularity Containers](#singularity-containers)
- [How to Perform Experiments](#how-to-perform-experiments)
  - [CFD Simulations](#cfd-simulations)
  - [Data for Deep Learning](#data-for-deep-learning)
  - [Deep Learning](#deep-learning)
  - [EnKF (Baseline Model)](#enkf-baseline-model)
  - [Analysis](#analysis)
- [Cite](#cite)


## Setup

- Basically, the Singularity containers were used for experiments.
- The Docker containers have the same environments as in the Singularity containers.
- `tsubame` means the super-computer at Tokyo Institute of Technology ([webpage](https://www.t3.gsic.titech.ac.jp/en)).
    - Singularity containers can be used on TSUBAME.

### Docker Containers

1. Install [Docker](https://docs.docker.com/get-started/)
1. Build docker containers: `$ docker compose build`
1. Start docker containers: `$ docker compose up -d`

### Singularity Containers

1. Install [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/quick_start.html)
1. Build Singularity containers:
    - `$ singularity build -f pytorch_local.sif ./singularity/pytorch/pytorch.def`
    - `$ singularity build -f pytorch_tsubame.sif ./singularity/pytorch_tsubame/pytorch_tsubame.def`
1. Start singularity containers:
    - The following command is for local environments

```sh
$ singularity exec --nv --env PYTHONPATH="./pytorch" \
    pytorch_local.sif jupyter lab \
    --no-browser --ip=0.0.0.0 --allow-root --LabApp.token='' --port=8888
```

## How to Perform Experiments

- The Singularity container, `pytorch_local.sif`, is used in the following experiments.
- On [TSUBAME](https://www.t3.gsic.titech.ac.jp/en), the same code was run using `pytorch_tsubame.sif`.

### CFD Simulations

1. Set the preferences for simulations
   - Specify root directory path and seed indices in [the shell script](./pytorch/script/shell/simulate_cfd_jet.sh), which performs [the pyton script](./pytorch/script/python/simulate_cfd_jet.py) in the Singularity container.
   - [This pyton script](./pytorch/script/python/simulate_cfd_jet.py) conducts CFD simulations using batch calculations, where the batch size is `20` (i.e., `N_ENSEMBLES = 20`).
2. Run [the shell script](./pytorch/script/shell/simulate_cfd_jet.sh): `$ ./pytorch/script/shell/simulate_cfd_jet.sh`
3. Confirm the simulation data exist in `./data/pytorch/CFD/jet01`.

### Data for Deep Learning

### Deep Learning

### EnKF (Baseline Model)

### Analysis


## Cite

```
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