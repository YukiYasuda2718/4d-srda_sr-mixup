#!/bin/bash

# Specify your root directory of this repository in the Singularity container.
ROOT_DIR= 

IMAGE_PATH="${ROOT_DIR}/pytorch_local.sif"
SCRIPT_PATH="${ROOT_DIR}/pytorch/script/python/train_ml_model.py"

# Change config path if necessary.
CONFIG_PATH="${ROOT_DIR}/pytorch/config/paper_experiment/lt4og12_on1e-01_ep1000_lr1e-04_scT_muT_a02_b02_sd221958.yml"

echo "image path = ${IMAGE_PATH}"
echo "script path = ${SCRIPT_PATH}"
echo "config path = ${CONFIG_PATH}"

singularity exec \
  --nv \
  --env PYTHONPATH=$ROOT_DIR/pytorch \
  ${IMAGE_PATH} python3 ${SCRIPT_PATH} --config_path ${CONFIG_PATH}
