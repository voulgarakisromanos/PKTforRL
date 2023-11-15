using Flux
using Random

include("../algorithms/TwinDelayedDDPGfromDemos.jl")

function create_actor(visual::Bool, rng::AbstractRNG, state_size::Int, action_size::Int)
    init = glorot_uniform(rng)
    if visual
        gpu(Chain(
            Conv((3, 3), 3 => 32, relu; stride=1, pad=0, init=init, bias=false),
            Conv((3, 3), 32 => 32, relu; stride=1, pad=0, init=init, bias=false),
            MaxPool((4, 4)),
            Conv((3, 3), 32 => 64, relu; stride=1, pad=0, init=init, bias=false),
            Conv((3, 3), 64 => 64, relu; stride=1, pad=0, init=init, bias=false),
            MaxPool((4, 4)),
            Flux.flatten,
            Dense(256, 60, relu; init=init),
            Dense(60, 60, relu; init=init),
            Dense(60, action_size, tanh; init=init),
        ))
    else
        gpu(Chain(
            Dense(state_size, 60, relu; init=init),
            Dense(60, 60, relu; init=init),
            Dense(60, action_size, tanh; init=init),
        ))
    end
end

function create_critic_model(
    visual::Bool, rng::AbstractRNG, state_size::Int, action_size::Int
)
    init = glorot_uniform(rng)
    if visual
        gpu(Chain(
            CombineActionImageEmbedding(
                Chain(
                    Conv((3, 3), 3 => 32, relu; stride=1, pad=0, init=init, bias=false),
                    Conv((3, 3), 32 => 32, relu; stride=1, pad=0, init=init, bias=false),
                    MaxPool((4, 4)),
                    Conv((3, 3), 32 => 64, relu; stride=1, pad=0, init=init, bias=false),
                    Conv((3, 3), 64 => 64, relu; stride=1, pad=0, init=init, bias=false),
                    MaxPool((4, 4)),
                    Flux.flatten,
                ),
                vcat,
            ),
            Dense(256 + action_size, 60, relu; init=init),
            Dense(60, 60, relu; init=init),
            Dense(60, 1; init=init),
        ))
    else
        gpu(Chain(
            Dense(state_size + action_size, 30, relu; init=init),
            Dense(30, 30, relu; init=init),
            Dense(30, 1; init=init),
        ))
    end
end

function create_critic(visual::Bool, rng::AbstractRNG, state_size::Int, action_size::Int)
    return TwinDelayedDDPGCritic{visual}([
        create_critic_model(visual, rng, state_size, action_size),
        create_critic_model(visual, rng, state_size, action_size),
    ])
end
