#!/bin/bash
#SBATCH --job-name=gather
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=64G   # memory per cpu-core

julia --project=. initialise_env.jl
julia --project=. src/scripts/gather_visual_demos.jl
