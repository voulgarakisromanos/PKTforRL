using Flux
include("../algorithms/TwinDelayedDDPGfromDemos.jl")

create_actor() = Chain(
        Conv((3, 3), 3 => 32, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
        Conv((3, 3), 32 => 32, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
        MaxPool((4, 4)),
        Conv((3, 3), 32 => 64, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
        Conv((3, 3), 64 => 64, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
        MaxPool((4, 4)),
        Flux.flatten,
        Dense(256, 30, relu; init = init),
        Dense(30, 30, relu; init = init),
        Dense(30, na, tanh; init = init),
) |> gpu

function create_critic_model()
    Chain( 
    CombineActionImageEmbedding(Chain(
        Conv((3, 3), 3 => 32, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
        Conv((3, 3), 32 => 32, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
        MaxPool((4, 4)),
        Conv((3, 3), 32 => 64, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
        Conv((3, 3), 64 => 64, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
        MaxPool((4, 4)),
        Flux.flatten
        ), vcat),
        Dense(256 + na, 30, relu; init = init),
        Dense(30, 30, relu; init = init),
        Dense(30, 1; init = init),
    )
end |> gpu

# create_actor() = Chain(
#     Dense(ns, 30, relu; init = init),
#     Dense(30, 30, relu; init = init),
#     Dense(30, na, tanh; init = init),
# ) |> gpu

# create_critic_model() = Chain(
#     Dense(ns + 1, 30, relu; init = init),
#     Dense(30, 30, relu; init = init),
#     Dense(30, 1; init = init),
# ) |> gpu

create_critic() = TwinDelayedDDPGCritic{visual}([create_critic_model(), create_critic_model()])