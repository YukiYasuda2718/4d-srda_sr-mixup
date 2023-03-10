# 4d-srda_sr-mixup <!-- omit in toc -->

[![license](https://img.shields.io/badge/license-CC%20BY--NC--SA-informational)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) [![reference](https://img.shields.io/badge/reference-arXiv-important)](https://arxiv.org/abs/2212.03656)  [![pytorch](https://img.shields.io/badge/PyTorch-1.11.0-informational)](https://pytorch.org/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7608394.svg)](https://doi.org/10.5281/zenodo.7608394)

This repository contains the source code used in *Spatio-Temporal Super-Resolution Data Assimilation (SRDA) Utilizing Deep Neural Networks with Domain Generalization Technique Toward Four-Dimensional SRDA* ([arXiv](https://arxiv.org/abs/2212.03656)).

- [Setup](#setup)
  - [Docker Containers](#docker-containers)
  - [Singularity Containers](#singularity-containers)
- [How to Perform Experiments](#how-to-perform-experiments)
  - [CFD Simulations](#cfd-simulations)
  - [Data for Deep Learning](#data-for-deep-learning)
  - [Deep Learning](#deep-learning)
  - [Evaluation of Trained Models](#evaluation-of-trained-models)
  - [Comparison with EnKF (Baseline Model)](#comparison-with-enkf-baseline-model)
- [Cite](#cite)

## Setup

- Basically, the Singularity containers were used for experiments.
- At least, 1 GPU board is required.
- Notes
  - The Docker containers have the same environments as in the Singularity containers.
  - `tsubame` means the super-computer at Tokyo Institute of Technology ([webpage](https://www.t3.gsic.titech.ac.jp/en)).
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

1. Set the preferences for CFD simulations
   - Specify root directory path and seed indices in [`simulate_cfd_jet.sh`](./pytorch/script/shell/simulate_cfd_jet.sh), which performs [`simulate_cfd_jet.py`](./pytorch/script/python/simulate_cfd_jet.py) in the Singularity container.
   - [`simulate_cfd_jet.py`](./pytorch/script/python/simulate_cfd_jet.py) conducts CFD simulations using batch calculations, where the batch size is `20` (i.e., `N_ENSEMBLES = 20`).
2. Run [`simulate_cfd_jet.sh`](./pytorch/script/shell/simulate_cfd_jet.sh): `$ ./pytorch/script/shell/simulate_cfd_jet.sh`
3. Confirm the simulation data exist in `./data/pytorch/CFD/jet01`.

### Data for Deep Learning

1. Run the Singularity container using `pytorch_local.sif` as described above.
2. Split the CFD simulation data by running [`split_data_for_io_latency.ipynb`](./pytorch/notebook/paper_experiment/split_data_for_io_latency.ipynb)
   - The IO latency will be lower because each simulation data (`.npy`) contains 20 simulation results.
   - Splitting into 20 files makes the latency higher.
3. Confirm the split simulation data exist in `./data/pytorch/CFD/jet02`.
4. Check `dataset` and `dataloader` by running [`check_dataset_dataloader.ipynb`](./pytorch/notebook/paper_experiment/check_dataset_dataloader.ipynb)

### Deep Learning

1. Set the preferences for deep learning
   - Specify root directory path and config path in [train_ml_model.sh](./pytorch/script/shell/train_ml_model.sh).
   - Configurations are stored in [config dir](./pytorch/config/paper_experiment/).
2. Perform deep learning: `$ ./pytorch/script/shell/train_ml_model.sh`
3. Repeat steps 1 and 2 until the experiments of all configurations are done.

### Evaluation of Trained Models

1. Evaluate trained models using SRDA (repeating feedback cycles) by running [evaluate_ml_model_using_srda.ipynb](./pytorch/notebook/paper_experiment/evaluate_ml_model_using_srda.ipynb)
2. Evaluate trained models using test dataloaders (without feedback cycles) by running [evaluate_ml_model_using_testdataset.ipynb](./pytorch/notebook/paper_experiment/evaluate_ml_model_using_testdataset.ipynb)

### Comparison with EnKF (Baseline Model)

1. Perform CFD simulations with EnKF by running [perform_enkf.ipynb](./pytorch/notebook/paper_experiment/perform_enkf.ipynb).
2. Compare the obtained results with those of SRDA by running [compare_srda_enkf.ipynb](./pytorch/notebook/paper_experiment/compare_srda_enkf.ipynb).

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
