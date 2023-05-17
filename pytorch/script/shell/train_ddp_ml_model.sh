#!/bin/bash

# Specify your root directory of this repository in the Singularity container.
ROOT_DIR=$(pwd)

IMAGE_PATH="${ROOT_DIR}/pytorch_local.sif"
SCRIPT_PATH="${ROOT_DIR}/pytorch/script/python/train_ddp_ml_model.py"

# Change config path if necessary.
CONFIG_PATH="${ROOT_DIR}/pytorch/config/perform_ST_SRDA/lt4og08_on1e-01_ep1000_lr1e-04_scT_bT_muT_a02_b02_sd221958.yml"

echo "image path = ${IMAGE_PATH}"
echo "script path = ${SCRIPT_PATH}"
echo "config path = ${CONFIG_PATH}"

# Change world size if necessary.
singularity exec \
  --nv \
  --env PYTHONPATH=$ROOT_DIR/pytorch \
  ${IMAGE_PATH} python3 ${SCRIPT_PATH} --config_path ${CONFIG_PATH} --world_size 1
