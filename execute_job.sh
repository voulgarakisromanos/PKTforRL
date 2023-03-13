julia --project=. initialise_env.jl 
julia --project=. src/scripts/gather_visual_demos.jl
julia --project=. src/scripts/train_student.jl
