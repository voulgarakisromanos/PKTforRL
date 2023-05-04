#!/bin/bash
#SBATCH --job-name=myjob        # Job name
#SBATCH --array=1-8           # Job array with 4 tasks
#SBATCH --time=1-00:00:00      # Maximum time for each task (1 day)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G   # memory
#SBATCH --cpus-per-task=10
#SBATCH --output=job_%A_%a.out  # Output file name for each task
#SBATCH --error=job_%A_%a.err   # Error file name for each task

# Define an array variable with the commands
commands=(
    "julia --project=. src/scripts/train_student.jl --env Lift --run_name Lift_linear_mse_rep_001_no_1 --similarity_function linear --pre_steps 0 --repr_weight 0.01 --total_steps 30000"
    "julia --project=. src/scripts/train_student.jl --env Lift --run_name Lift_linear_mse_rep_001_no_2 --similarity_function linear --pre_steps 0 --repr_weight 0.01 --total_steps 30000"
    "julia --project=. src/scripts/train_student.jl --env Lift --run_name Lift_linear_mse_rep_001_no_3 --similarity_function linear --pre_steps 0 --repr_weight 0.01 --total_steps 30000"
    "julia --project=. src/scripts/train_student.jl --env Lift --run_name Lift_linear_mse_rep_001_no_4 --similarity_function linear --pre_steps 0 --repr_weight 0.01 --total_steps 30000"
    "julia --project=. src/scripts/train_student.jl --env Lift --run_name Lift_linear_mse_rep_10_no_1 --similarity_function linear --pre_steps 0 --repr_weight 10.0 --total_steps 30000"
    "julia --project=. src/scripts/train_student.jl --env Lift --run_name Lift_linear_mse_rep_10_no_2 --similarity_function linear --pre_steps 0 --repr_weight 10.0 --total_steps 30000"
    "julia --project=. src/scripts/train_student.jl --env Lift --run_name Lift_linear_mse_rep_10_no_3 --similarity_function linear --pre_steps 0 --repr_weight 10.0 --total_steps 30000"
    "julia --project=. src/scripts/train_student.jl --env Lift --run_name Lift_linear_mse_rep_10_no_4 --similarity_function linear --pre_steps 0 --repr_weight 10.0 --total_steps 30000"
   )

# Select the command based on the task ID
command=${commands[$SLURM_ARRAY_TASK_ID-1]}

# Run the selected command
$command
