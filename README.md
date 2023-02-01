# 4d-srda_sr-mixup <!-- omit in toc -->

[![reference](https://img.shields.io/badge/reference-arXiv-orange)](https://arxiv.org/abs/2212.03656) [![license](https://img.shields.io/badge/license-CC%20BY--NC--SA-blue)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt)

This repository contains the source code used in *Spatio-Temporal Super-Resolution Data Assimilation (SRDA) Utilizing Deep Neural Networks with Domain Generalization Technique Toward Four-Dimensional SRDA* ([arXiv](https://arxiv.org/abs/2212.03656)).

- [Setup](#setup)
  - [Docker Containers](#docker-containers)
  - [Singularity Containers](#singularity-containers)
- [How to Perform Experiments](#how-to-perform-experiments)


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