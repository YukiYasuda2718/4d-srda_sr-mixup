#!/bin/bash

ROOT_DIR= # Specify your root directory of this repository.

IMAGE_PATH="${ROOT_DIR}/pytorch_local.sif"
SCRIPT_PATH="${ROOT_DIR}/pytorch/script/python/perform_enkf_jet"
CONFIG_PATH="${ROOT_DIR}/pytorch/config/enkf/e100_sysstd_1e+00_sysdx_1e+00_locdx_1e+00_train0200.yml"

echo "image path = ${IMAGE_PATH}"
echo "script path = ${SCRIPT_PATH}"
echo "config path = ${CONFIG_PATH}"

singularity exec \
  --nv \
  --env PYTHONPATH=$ROOT_DIR/pytorch \
  $IMAGE_PATH python3 $SCRIPT_PATH --config_path ${CONFIG_PATH}