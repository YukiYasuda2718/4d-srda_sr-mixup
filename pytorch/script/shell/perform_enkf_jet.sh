#!/bin/bash

ROOT_DIR= # Specify your root directory of this repository.

IMAGE_PATH="${ROOT_DIR}/pytorch_local.sif"
SCRIPT_PATH="${ROOT_DIR}/pytorch/script/python/perform_enkf_jet"
CONFIG_PATH="${ROOT_DIR}/pytorch/config/enkf_01/e100_sysstd_3e-01_sysdx_3e-01_locdx_3e-01_train0100.yml"

echo "image path = ${IMAGE_PATH}"
echo "script path = ${SCRIPT_PATH}"
echo "config path = ${CONFIG_PATH}"

singularity exec \
  --nv \
  --env PYTHONPATH=$ROOT_DIR/pytorch \
  $IMAGE_PATH python3 $SCRIPT_PATH --config_path ${CONFIG_PATH}