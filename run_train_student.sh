#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=64G   # memory per cpu-core

julia --project=. initialise_env.jl
julia --project=. src/scripts/train_student.jl
