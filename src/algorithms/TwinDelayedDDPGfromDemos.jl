struct TwinDelayedDDPGCritic{visual}
    critic_nets::Vector{Flux.Chain}
end

Flux.@functor TwinDelayedDDPGCritic
function (c::TwinDelayedDDPGCritic{false})(s, a)
    return (inp = vcat(s, a); (c.critic_nets[1](inp), c.critic_nets[2](inp)))
end
function (c::TwinDelayedDDPGCritic{false})(s, a, critic_selection::Int)
    return (inp = vcat(s, a); (c.critic_nets[critic_selection](inp)))
end
function (c::TwinDelayedDDPGCritic{true})(s, a)
    return (inp = (s, a); (c.critic_nets[1](inp), c.critic_nets[2](inp)))
end
function (c::TwinDelayedDDPGCritic{true})(s, a, critic_selection::Int)
    return (inp = (s, a); (c.critic_nets[critic_selection](inp)))
end

mutable struct LossStruct{T}
    critic_loss::T
    critic_q_loss::T
    critic_l2_loss::T
    actor_loss::T
    actor_q_loss::T
    actor_bc_loss::T
    actor_l2_loss::T
    distill_loss::T
end

@concrete struct TwinDelayedDDPGPolicy <: AbstractPolicy
    behavior_actor
    behavior_critic
    target_actor
    target_critic
    teacher
    γ
    ρ
    batch_size
    start_steps
    start_policy
    pretraining_steps
    update_freq
    policy_freq
    target_act_limit
    target_act_noise
    act_limit
    act_noise
    step_counter
    rng
    policy_update_counter
    q_bc_weight
    critic_l2_weight
    actor_l2_weight
    distill_weight
    similarity_function
    distill_layer
    loss_struct
end

"""
TwinDelayedDDPGPolicy(;kwargs...)

# Keyword arguments

- `behavior_actor`,
- `behavior_critic`,
- `target_actor`,
- `target_critic`,
- `teacher`,
- `γ = 0.99f0`,
- `ρ = 0.995f0`,
- `batch_size = 256`,
- `start_steps`,
- `start_policy`,
- `pretraining_steps`,
- `update_freq`, # how many env interactions are needed before updating
- `policy_freq`, # how many critic updates are needed before updating the policy
- `target_act_limit`, # noise added to actor target
- `target_act_noise`, # noise added to actor target
- `act_limit`, # noise added when outputing action
- `act_noise`, # noise added when outputing action
- `step_counter`,
- `rng`,
- `policy_update_counter`,
- `q_bc_weight`,
- `critic_l2_weight`,
- `actor_l2_weight`,
- `distill_weight`,
- `similarity_function`,
- `distill_layer`,
- `for logging`,
- `loss_struct` # for logging purposes

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
    rng=Random.GLOBAL_RNG,
    q_bc_weight=1.0,
    critic_l2_weight=1.0,
    actor_l2_weight=1.0,
    distill_weight=1.0,
    similarity_function,
    distill_layer,
)
    copyto!(behavior_actor, target_actor) 
    copyto!(behavior_critic, target_critic)
    return TwinDelayedDDPGPolicy(
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
        Ref(0),
        rng,
        Ref(1),
        q_bc_weight,
        critic_l2_weight,
        actor_l2_weight,
        distill_weight,
        similarity_function,
        distill_layer,
        LossStruct(zeros(Float32, 8)...),
    )
end

function (p::TwinDelayedDDPGPolicy)(env)
    if p.step_counter[] <= p.start_steps
        p.start_policy(env)
    else
        D = device(p.behavior_actor)
        s = state(env)
        s = Flux.unsqueeze(s, ndims(s) + 1)
        action = send_to_host(vec(p.behavior_actor(send_to_device(D, s))))
        clamp.(
            action .+ randn(p.rng, length(action)) .* p.act_noise, -p.act_limit, p.act_limit
        )
    end
end

function training_step(p::TwinDelayedDDPGPolicy, traj::CombinedTrajectory)
    p.step_counter[] % p.update_freq == 0 || return nothing
    length(traj) > (p.batch_size) || return nothing

    demo_sample_length = Int(round(traj.ratio * p.batch_size))
    main_sample_length = p.batch_size - demo_sample_length

    demo_sampler = BatchSampler{SGARTSG}(demo_sample_length)
    main_sampler = BatchSampler{SGARTSG}(main_sample_length)

    _, main_batch = main_sampler(traj.main_trajectory)
    _, demo_batch = demo_sampler(traj.demo_trajectory)

    full_batch = combine_named_tuples(main_batch, demo_batch)

    return update!(p, full_batch)
end

function pretraining_step(p::TwinDelayedDDPGPolicy, traj::CombinedTrajectory)
    sampler = BatchSampler{SGARTSG}(p.batch_size)
    _, demo_batch = sampler(traj.demo_trajectory)
    return update!(p, demo_batch)
end

function RLBase.update!(
    p::TwinDelayedDDPGPolicy, traj::CombinedTrajectory, ::AbstractEnv, ::PreActStage
)
    p.step_counter[] += 1

    if p.step_counter[] > p.pretraining_steps  # Sampling mix of demonstrations and acquired transitions
        training_step(p, traj)
    elseif p.step_counter[] <= p.pretraining_steps # Pretraining
        pretraining_step(p, traj)
    end
end

"""
RLBase.update!(p::TwinDelayedDDPGPolicy, batch::NamedTuple)

# Keyword arguments
- `p::TwinDelayedDDPGPolicy`,
- `batch::NamedTuple`
"""
function RLBase.update!(p::TwinDelayedDDPGPolicy, batch::NamedTuple)
    to_device(x) = send_to_device(device(p.behavior_actor), x)
    s, gt, a, r, t, s′, gt′ = to_device(batch)

    actor = p.behavior_actor
    critic = p.behavior_critic

    teacher_actor = gpu(p.teacher.actor)

    target_noise = to_device(
        clamp.(
            randn(p.rng, Float32, size(a)[1], p.batch_size) .* p.target_act_noise,
            -p.target_act_limit,
            p.target_act_limit,
        ),
    )

    a′ = clamp.(p.target_actor(s′) + target_noise, -p.act_limit, p.act_limit)
    q_1′, q_2′ = p.target_critic(s′, a′)
    y = r .+ p.γ .* (1 .- t) .* (vec(min.(q_1′, q_2′)))

    gs1 = gradient(Flux.params(critic)) do
        q1, q2 = critic(s, a)
        q_loss = Flux.mse(vec(q1), y) + Flux.mse(vec(q2), y)
        l2_loss = sum(x -> sum(abs2, x) / 2, Flux.params(critic))
        loss = q_loss + p.critic_l2_weight * l2_loss
        Flux.ignore() do
            p.loss_struct.critic_loss = loss
            p.loss_struct.critic_q_loss = q_loss
            p.loss_struct.critic_l2_loss = l2_loss
        end
        loss
    end

    update!(critic, gs1)

    if p.policy_update_counter[] % p.policy_freq == 0
        gs2 = gradient(Flux.params(actor)) do
            actions = actor(s)
            activations = actor.model[1:(end - p.distill_layer)](s)
            q_loss = -mean(critic(s, actions, 1))
            q_scale = mean(abs.(critic(s, a, 1)))
            bc_loss = mean((actions .- teacher_actor(gt)) .^ 2)
            distill_loss = p.similarity_function(
                activations, teacher_actor[1:(end - p.distill_layer)](gt)
            )
            l2_loss = sum(x -> sum(abs2, x) / 2, Flux.params(actor))
            λ = p.q_bc_weight / (q_scale)
            loss =
                λ * q_loss +
                bc_loss +
                p.actor_l2_weight * l2_loss +
                p.distill_weight * distill_loss
            Flux.ignore() do
                p.loss_struct.actor_loss = loss
                p.loss_struct.actor_q_loss = q_loss
                p.loss_struct.actor_bc_loss = bc_loss
                p.loss_struct.actor_l2_loss = l2_loss
                p.loss_struct.distill_loss = distill_loss
            end
            loss
        end

        update!(actor, gs2)

        for (dest, src) in zip(
            Flux.params([p.target_actor, p.target_critic]), Flux.params([actor, critic])
        )
            dest .= p.ρ .* dest .+ (1 - p.ρ) .* src
        end
        p.policy_update_counter[] = 1
    end
    return p.policy_update_counter[] += 1
end

function pretrain(agent::AbstractPolicy, steps::Int)
    for step in 1:steps
        pretraining_step(agent.policy, agent.trajectory)
    end
end

function pretrain_run(
    agent::AbstractPolicy,
    env::AbstractEnv,
    stop_condition=StopAfterEpisode(1),
    hook=EmptyHook(),
)
    while agent.policy.step_counter[] < agent.policy.pretraining_steps
        update!(agent.policy, agent.trajectory, env, PRE_ACT_STAGE)
        hook(POST_ACT_STAGE, agent, env)
    end
    return RLCore._run(agent, env, stop_condition, hook)
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    ::TwinDelayedDDPGPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    if length(trajectory) > 0
        pop!(trajectory[:state])
        pop!(trajectory[:action])
        pop!(trajectory[:groundtruth])
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::TwinDelayedDDPGPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    s = state(env)
    push!(trajectory[:state], s)
    push!(trajectory[:action], action)
    return push!(trajectory[:groundtruth], get_groundtruth_state(env))
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::TwinDelayedDDPGPolicy,
    env::AbstractEnv,
    ::PostEpisodeStage,
)
    s = state(env)
    A = action_space(env)
    a = RLCore.get_dummy_action(A)

    push!(trajectory[:state], s)
    push!(trajectory[:action], a)
    return push!(trajectory[:groundtruth], get_groundtruth_state(env))
end
