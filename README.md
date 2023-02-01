# 4d-srda_sr-mixup

## Setup

### Docker services

1. Install [Docker](https://docs.docker.com/get-started/)
1. Build docker containers: `$ docker compose build`
1. Start docker containers: `$ docker compose up -d`

### Singularity services

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