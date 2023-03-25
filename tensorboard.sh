#!/bin/bash
module load gcc miniconda3
source $CONDA_PROFILE/conda.sh
conda activate julia_env
tensorboard --logdir=kd-rlfd/logs/ --port=16006