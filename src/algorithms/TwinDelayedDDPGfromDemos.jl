using Statistics
using Random

struct TwinDelayedDDPGCritic{visual}
    critic_nets::Vector{Flux.Chain}
end

Flux.@functor TwinDelayedDDPGCritic
(c::TwinDelayedDDPGCritic{false})(s, a) = (inp = vcat(s, a); (c.critic_nets[1](inp), c.critic_nets[2](inp)))
(c::TwinDelayedDDPGCritic{false})(s, a, critic_selection::Int) = (inp = vcat(s, a); (c.critic_nets[critic_selection](inp)))
(c::TwinDelayedDDPGCritic{true})(s, a) = (inp = (s, a); (c.critic_nets[1](inp), c.critic_nets[2](inp)))
(c::TwinDelayedDDPGCritic{true})(s, a, critic_selection::Int) = (inp = (s, a); (c.critic_nets[critic_selection](inp)))

mutable struct TwinDelayedDDPGPolicy{
    BA<:NeuralNetworkApproximator,
    BC<:NeuralNetworkApproximator,
    TA<:NeuralNetworkApproximator,
    TC<:NeuralNetworkApproximator,
    P,
    R<:AbstractRNG,
} <: AbstractPolicy

    behavior_actor::BA
    behavior_critic::BC
    target_actor::TA
    target_critic::TC
    teacher
    γ::Float32
    ρ::Float32
    batch_size::Int
    start_steps::Int
    start_policy::P
    pretraining_steps::Int
    update_freq::Int
    policy_freq::Int
    target_act_limit::Float64
    target_act_noise::Float64
    act_limit::Float64
    act_noise::Float64
    update_step::Int
    rng::R
    replay_counter::Int
    q_bc_weight::Float32
    critic_l2_weight::Float32
    actor_l2_weight::Float32
    representation_weight::Float32
    # for logging
    critic_loss::Float32
    critic_q_loss::Float32
    critic_l2_loss::Float32
    actor_loss::Float32
    actor_q_loss::Float32
    actor_bc_loss::Float32
    actor_l2_loss::Float32
    representation_loss::Float32
end

"""
TwinDelayedDDPGCritic(;kwargs...)

# Keyword arguments

- `behavior_actor`,
- `behavior_critic`,
- `target_actor`,
- `target_critic`,
- `start_policy`,
- `γ = 0.99f0`,
- `ρ = 0.995f0`,
- `batch_size = 32`,
- `start_steps = 10000`,
- `update_after = 1000`,
- `update_freq = 50`,
- `policy_freq = 2` # frequency in which the actor performs a gradient update_step and critic target is updated
- `target_act_limit = 1.0`, # noise added to actor target
- `target_act_noise = 0.1`, # noise added to actor target
- `act_limit = 1.0`, # noise added when outputing action
- `act_noise = 0.1`, # noise added when outputing action
- `update_step = 0`,
- `rng = Random.GLOBAL_RNG`,
"""
function TwinDelayedDDPGPolicy(;
    behavior_actor,
    behavior_critic,
    target_actor,
    target_critic,
    teacher,
    start_policy,
    γ=0.99f0,
    ρ=0.995f0,
    batch_size=64,
    start_steps=10000,
    pretraining_steps=1000,
    update_freq=50,
    policy_freq=2,
    target_act_limit=1.0,
    target_act_noise=0.1,
    act_limit=1.0,
    act_noise=0.1,
    update_step=0,
    rng=Random.GLOBAL_RNG,
    q_bc_weight=1.0,
    critic_l2_weight=1.0,
    actor_l2_weight=1.0,
    representation_weight=1.0
)
    copyto!(behavior_actor, target_actor)  # force sync
    copyto!(behavior_critic, target_critic)  # force sync
    TwinDelayedDDPGPolicy(
        behavior_actor,
        behavior_critic,
        target_actor,
        target_critic,
        teacher,
        γ,
        ρ,
        batch_size,
        start_steps,
        start_policy,
        pretraining_steps,
        update_freq,
        policy_freq,
        target_act_limit,
        target_act_noise,
        act_limit,
        act_noise,
        update_step,
        rng,
        1, # keep track of numbers of replay
        q_bc_weight,
        critic_l2_weight,
        actor_l2_weight,
        representation_weight,
        zeros(Float32, 8)...
    )
end

function (p::TwinDelayedDDPGPolicy)(env)
    if p.update_step <= p.start_steps
        p.start_policy(env)
    else
        D = device(p.behavior_actor)
        s = state(env)
        s = Flux.unsqueeze(s, ndims(s) + 1)
        action = p.behavior_actor(send_to_device(D, s)) |> vec |> send_to_host
        clamp.(action .+ randn(p.rng, length(action)) .* p.act_noise, -p.act_limit, p.act_limit)
    end
end

function training_step(p::TwinDelayedDDPGPolicy, traj::CombinedTrajectory)
    p.update_step % p.update_freq == 0 || return
    length(traj) > (p.batch_size) || return

    demo_sample_length = Int(round(traj.ratio * p.batch_size))
    main_sample_length = p.batch_size - demo_sample_length

    demo_sampler = BatchSampler{SGARTSG}(demo_sample_length)
    main_sampler = BatchSampler{SGARTSG}(main_sample_length)
    _, main_batch = main_sampler(traj.main_trajectory)
    _, demo_batch = demo_sampler(traj.demo_trajectory)
    full_batch = combine_named_tuples(main_batch, demo_batch)
    update!(p, full_batch, demo_batch)
end

function pretraining_step(p::TwinDelayedDDPGPolicy, traj::CombinedTrajectory)
    sampler = BatchSampler{SGARTSG}(p.batch_size)
    _, demo_batch = sampler(traj.demo_trajectory)
    update!(p, demo_batch, demo_batch)
end

function RLBase.update!(
    p::TwinDelayedDDPGPolicy,
    traj::CombinedTrajectory,
    ::AbstractEnv,
    ::PreActStage,
)
    p.update_step += 1

    if p.update_step > p.pretraining_steps  # Sampling mix of demonstrations and acquired transitions
        training_step(p, traj)
    elseif p.update_step <= p.pretraining_steps # Pretraining
        pretraining_step(p, traj)
    end
end

function RLBase.update!(p::TwinDelayedDDPGPolicy, batch::NamedTuple, demo_batch::NamedTuple)
    to_device(x) = send_to_device(device(p.behavior_actor), x)
    s, gt, a, r, t, s′, gt′ = to_device(batch)
    s_demo, gt_demo, a_demo, r_demo, t_demo, s′_demo, gt′_demo = to_device(demo_batch)

    actor = p.behavior_actor
    critic = p.behavior_critic

    target_noise =
        clamp.(
            randn(p.rng, Float32, size(a)[1], p.batch_size) .* p.target_act_noise,
            -p.target_act_limit,
            p.target_act_limit,
        ) |> to_device

    a′ = clamp.(p.target_actor(s′) + target_noise, -p.act_limit, p.act_limit)
    q_1′, q_2′ = p.target_critic(s′, a′)
    y = r .+ p.γ .* (1 .- t) .* (min.(q_1′, q_2′) |> vec)

    gs1 = gradient(Flux.params(critic)) do
        q1, q2 = critic(s, a)
        q_loss = Flux.mse(q1 |> vec, y) + Flux.mse(q2 |> vec, y)
        l2_loss = sum(x -> sum(abs2, x) / 2, Flux.params(critic))
        loss = q_loss + p.critic_l2_weight * l2_loss
        Flux.ignore() do
            p.critic_loss = loss
            p.critic_q_loss = q_loss
            p.critic_l2_loss = l2_loss
        end
        loss
    end

    update!(critic, gs1)
    
    if p.replay_counter % p.policy_freq == 0
        gs2 = gradient(Flux.params(actor)) do
            actions = actor(s)
            q_loss = -mean(critic(s, actions, 1))
            q_scale = mean(abs.(critic(s, a, 1)))
            bc_loss = mean((actor(s_demo) .- a_demo) .^ 2)
            l2_loss = sum(x -> sum(abs2, x) / 2, Flux.params(actor))
            representation_loss = cosine_similarity_loss(actions, p.teacher.actor(gt))
            λ = p.q_bc_weight / (q_scale)
            loss = λ * q_loss + bc_loss + p.actor_l2_weight * l2_loss + p.representation_weight * representation_loss
            Flux.ignore() do
                p.actor_loss = loss
                p.actor_q_loss = q_loss
                p.actor_bc_loss = bc_loss
                p.actor_l2_loss = l2_loss
                p.representation_loss = representation_loss
            end
            loss
        end

        update!(actor, gs2)

        for (dest, src) in zip(
            Flux.params([p.target_actor, p.target_critic]),
            Flux.params([actor, critic]),
        )
            dest .= p.ρ .* dest .+ (1 - p.ρ) .* src
        end
        p.replay_counter = 1
    end
    p.replay_counter += 1
end

function pretrain(agent::AbstractPolicy, steps::Int)
    for step = 1:steps
        pretraining_step(agent.policy, agent.trajectory)
    end
end

function pretrain_run(
    agent::AbstractPolicy,
    env::AbstractEnv,
    stop_condition=StopAfterEpisode(1),
    hook=EmptyHook(),
)   
    while agent.policy.update_step < agent.policy.pretraining_steps
        update!(agent.policy, agent.trajectory, env, PRE_ACT_STAGE)
        hook(POST_ACT_STAGE, agent, env)
    end
    RLCore._run(agent, env, stop_condition, hook)
end