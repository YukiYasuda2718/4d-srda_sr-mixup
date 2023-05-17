#!/bin/bash

# Specify your root directory of this repository in the Singularity container.
ROOT_DIR=$(pwd)

IMAGE_PATH="${ROOT_DIR}/pytorch_local.sif"
SCRIPT_PATH="${ROOT_DIR}/pytorch/script/python/simulate_cfd_jet_uhr.py"

echo "image path = ${IMAGE_PATH}"
echo "script path = ${SCRIPT_PATH}"

singularity exec \
  --nv \
  --env PYTHONPATH=$ROOT_DIR/pytorch \
  $IMAGE_PATH python3 $SCRIPT_PATH --device cuda