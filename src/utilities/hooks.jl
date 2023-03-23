using ReinforcementLearning

include("utils.jl")

const SGART = (:state, :groundtruth, :action, :reward, :terminal)
const SGARTSG = (:state, :groundtruth, :action, :reward, :terminal, :next_state, :next_groundtruth)

"""
Circular Array Trajectory with State-Groundtruth-Action-Reward traces.
"""
const CircularArraySGARTTrajectory = Trajectory{
    <:NamedTuple{
        SGART,
        <:Tuple{
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
        },
    },
}

CircularArraySGARTTrajectory(;
    capacity::Int,
    state = Int => (),
    groundtruth = Int = (),
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
) = merge(
    CircularArrayTrajectory(; capacity = capacity + 1, state = state, groundtruth = groundtruth, action = action),
    CircularArrayTrajectory(; capacity = capacity, reward = reward, terminal = terminal),
)


function fetch!(s::BatchSampler, t::CircularArraySGARTTrajectory, inds::Vector{Int})
    batch = NamedTuple{SGARTSG}((
        (consecutive_view(t[x], inds) for x in SGART)...,
        consecutive_view(t[:state], inds .+ 1),
        consecutive_view(t[:groundtruth], inds .+ 1),
    ))
    if isnothing(s.cache)
        s.cache = map(batch) do x
            convert(Array, x)
        end
    else
        map(s.cache, batch) do dest, src
            copyto!(dest, src)
        end
    end
end

Base.length(t::CircularArraySGARTTrajectory) = length(t[:reward])

mutable struct SampleTrajectory{include_groundtruth} <: AbstractHook
    t::AbstractTrajectory
end

function (hook::SampleTrajectory{false})(::PreActStage, agent, env, action)
    push!(hook.t, state=state(env), action=action)
end

function (hook::SampleTrajectory{true})(::PreActStage, agent, env, action)
    push!(hook.t, state=state(env), groundtruth=vcat(vec(env.proprioception_state), vec(env.object_state)), action=action)
end

function (hook::SampleTrajectory)(::PostActStage, agent, env)
    push!(hook.t, reward=reward(env), terminal=is_terminated(env))
end

mutable struct StateImageTransition <: AbstractHook
    t::AbstractTrajectory
end

function (hook::StateImageTransition)(::PreActStage, agent, env, action)
    push!(hook.t, state=env.grayscale, action=action[1])
end

function (hook::StateImageTransition)(::PostActStage, agent, env)
    push!(hook.t, reward=reward(env), terminal=is_terminated(env))
end

function tensorboard_hook(agent, tf_log_dir="logs/Lift"; save_checkpoints=false, save_frequency=20_000, agent_name="agents/visual/")
    lg = TBLogger(tf_log_dir, min_level = Logging.Info)
    total_reward_per_episode = TotalRewardPerEpisode()
    total_reward_per_episode.rewards = [0.0]
    hook = ComposedHook(
        total_reward_per_episode,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info  "losses" agent.policy.critic_loss agent.policy.critic_q_loss agent.policy.critic_l2_loss agent.policy.actor_loss agent.policy.actor_q_loss agent.policy.actor_bc_loss agent.policy.actor_l2_loss agent.policy.representation_loss
            end
            if t % save_frequency == 0
                save_agent(agent, string(agent_name,"_",string(t)))
            end 
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info  "reward" total_reward_per_episode.rewards[end]
            end
        end
    )
end