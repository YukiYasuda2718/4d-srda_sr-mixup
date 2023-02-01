#!/bin/bash

ROOT_DIR= # Specify your root directory of this repository.

IMAGE_PATH="${ROOT_DIR}/pytorch_local.sif"
SCRIPT_PATH="${ROOT_DIR}/pytorch/script/python/train_ml_model.py"

# Change config path if necessary.
CONFIG_PATH="${ROOT_DIR}/pytorch/config/paper_experiment/t05_ap04_soT_bF_lr1e-4.yml"

echo "image path = ${IMAGE_PATH}"
echo "script path = ${SCRIPT_PATH}"
echo "config path = ${CONFIG_PATH}"

singularity exec \
  --nv \
  --env PYTHONPATH=$ROOT_DIR/pytorch \
  ${IMAGE_PATH} python3 ${SCRIPT_PATH} --config_path ${CONFIG_PATH}
