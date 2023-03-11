using Flux
include("../algorithms/TwinDelayedDDPGfromDemos.jl")

function create_actor(visual::Bool)
    if visual 
        Chain(
            Conv((3, 3), 3 => 32, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
            Conv((3, 3), 32 => 32, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
            MaxPool((4, 4)),
            Conv((3, 3), 32 => 64, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
            Conv((3, 3), 64 => 64, relu, stride=1, pad=0, init=glorot_uniform(rng), bias=false),
            MaxPool((4, 4)),
            Flux.flatten,
            Dense(256, 60, relu; init = init),
            Dense(60, 60, relu; init = init),
            Dense(60, na, tanh; init = init),
        ) |> gpu
    else
        Chain(
            Dense(ns, 60, relu; init = init),
            Dense(60, 60, relu; init = init),
            Dense(60, na, tanh; init = init),
        )|> gpu
    end
end

function create_critic_model(visual)
    if visual
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
        )|> gpu
    else
        Chain(
        Dense(ns + 1, 30, relu; init = init),
        Dense(30, 30, relu; init = init),
        Dense(30, 1; init = init),
        )|> gpu
    end
end

create_critic(visual) = TwinDelayedDDPGCritic{visual}([create_critic_model(visual), create_critic_model(visual)])