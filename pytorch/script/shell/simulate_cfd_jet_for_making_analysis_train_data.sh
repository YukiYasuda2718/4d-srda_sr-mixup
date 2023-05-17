#!/bin/bash

 # Specify your root directory of this repository in the Singularity container.
ROOT_DIR=$(pwd)

IMAGE_PATH="${ROOT_DIR}/pytorch_local.sif"
SCRIPT_PATH="${ROOT_DIR}/pytorch/script/python/simulate_cfd_jet_for_making_analysis_train_data.py"

# Change seed indices if necessary.
I_SEED_START=0
I_SEED_END=9 # in paper, we set 249

echo "image path = ${IMAGE_PATH}"
echo "script path = ${SCRIPT_PATH}"
echo "indices of seeds are from ${I_SEED_START} to ${I_SEED_END}"

singularity exec \
  --nv \
  --env PYTHONPATH=$ROOT_DIR/pytorch \
  $IMAGE_PATH python3 $SCRIPT_PATH \
  --i_seed_start $I_SEED_START --i_seed_end $I_SEED_END