using Random
using CUDA
using StableRNGs
using ReinforcementLearning
using BSON
using Flux
using Robosuite
using IntervalSets
using TensorBoardLogger
using Logging
using CircularArrayBuffers

function tensorboard_training_hook(agent, tf_log_dir="logs/TwoArmPegInHole")
    lg = TBLogger(tf_log_dir, min_level = Logging.Info)
    total_reward_per_episode = TotalRewardPerEpisode()
    total_reward_per_episode.rewards = [0.0]
    hook = ComposedHook(
            total_reward_per_episode,
            DoEveryNStep() do t, agent, env
                with_logger(lg) do
                    @info  "losses" critic_loss = agent.policy.critic_loss  actor_loss = agent.policy.actor_loss 
                end
            end,
            DoEveryNEpisode() do t, agent, env
                with_logger(lg) do
                    @info  "reward" total_reward_per_episode.rewards[end]
                end
            end,
            SuccessRateHook(success_criterion=()->env.success, logger=lg)
        )
end

include("../models/network_definitions.jl")
include("../algorithms/TwinDelayedDDPGBase.jl")
include("../utilities/CombinedTrajectory.jl")
include("../utilities/hooks.jl")

robots = ("Panda", "Panda")
env = RoboticEnv(name="TwoArmPegInHole", robots=robots, T=Float32, controller="OSC_POSE", enable_visual=false, show=false, horizon=200)

rng = StableRNG(123)
init = glorot_uniform(rng)
na, ns = size(action_space(env))[1], size(state_space(env))[1]

create_actor() = Chain(
    Dense(ns, 60, relu; init = init),
    Dense(60, 60, relu; init=init),
    Dense(60, na, tanh; init = init),
) |> gpu

create_critic_model() = Chain(
    Dense(ns + na, 60, relu; init = init),
    Dense(60, 60, relu; init=init),
    Dense(60, 1; init = init),
    vec
) |> gpu

create_critic() = TwinDelayedDDPGBaseCritic(create_critic_model(), create_critic_model())

agent = Agent(
    policy = TwinDelayedDDPGBasePolicy(
        behavior_actor = NeuralNetworkApproximator(
            model = create_actor(),
            optimizer = ADAM(1e-4),
        ),
        behavior_critic = NeuralNetworkApproximator(
            model = create_critic(),
            optimizer = ADAM(1e-4),
        ),
        target_actor = NeuralNetworkApproximator(
            model = create_actor(),
            optimizer = ADAM(1e-4),
        ),
        target_critic = NeuralNetworkApproximator(
            model = create_critic(),
            optimizer = ADAM(1e-4),
        ),
        γ = 0.99f0,
        ρ = 0.95f0,
        batch_size = 2048,
        start_steps = 1000,
        start_policy = RandomPolicy(Space([-1.0..1.0 for i=1:na]); rng = rng),
        update_after = 1000,
        update_freq = 1,
        policy_freq = 2,
        target_act_limit = 1.0,
        target_act_noise = 0.1,
        act_limit = 1.0,
        act_noise = 0.1,
        rng = rng,
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 3_000_000,
        state = Vector{Float32} => (ns,),
        action = Float32 => (na,),
    ),
)

stop_condition = StopAfterStep(2_000_000, is_show_progress=!haskey(ENV, "CI"));

hook = tensorboard_training_hook(agent)

run(agent, env, stop_condition, hook)

agent = agent |> cpu
agent = (actor = agent.policy.behavior_actor.model, critic = agent.policy.behavior_critic.model.critic_1)
BSON.@save "agents/groundtruth/TwoArmPegInHole" agent
