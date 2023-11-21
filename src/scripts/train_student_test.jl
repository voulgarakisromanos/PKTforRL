include("../utilities/training_dependencies.jl")

function main()
    env_name = "Door"
    agent_name = "Door"
    pretraining_steps = 0
    representation_weight = 0.1f0
    q_bc_weight = 2.5f0
    run_name = "run_1"
    total_steps = 100_000
    similarity_function = (x, y) -> linear_similarity_loss(x, y)
    distill_layer = 1 # beginning from the last

    image_size = 64
    frame_size = 3
    visual = true

    robots = "Panda"

    rng = StableRNG(123)
    env = RoboticEnv(;
        name=env_name,
        robots=robots,
        T=Float32,
        controller="OSC_POSE",
        enable_visual=visual,
        show=false,
        horizon=200,
        image_size=image_size,
        stop_when_done=true,
    )

    na = env.degrees_of_freedom
    ns = size(vcat(vec(env.proprioception_state), vec(env.object_state)))[1]

    BSON.@load string("datasets/", env_name, "_demo.bson") dataset
    BSON.@load string("agents/groundtruth/", agent_name) agent

    demo_trajectory = dataset

    trajectory = CombinedTrajectory(
        CircularArraySGARTTrajectory(;
            capacity=300_000,
            state=Vector{Float32} => (image_size, image_size, frame_size),
            groundtruth=Vector{Float32} =>
                (size(vcat(vec(env.proprioception_state), vec(env.object_state)))[1]),
            action=Vector{Float32} => (na,),
        ),
        demo_trajectory,
        0.25,
    )

    agent = Agent(;
        policy=TwinDelayedDDPGPolicy(;
            behavior_actor=NeuralNetworkApproximator(;
                model=create_actor(visual, rng, ns, na), optimizer=ADAM(1e-3)
            ),
            behavior_critic=NeuralNetworkApproximator(;
                model=create_critic(visual, rng, ns, na), optimizer=ADAM(1e-3)
            ),
            target_actor=NeuralNetworkApproximator(;
                model=create_actor(visual, rng, ns, na), optimizer=ADAM(1e-3)
            ),
            target_critic=NeuralNetworkApproximator(;
                model=create_critic(visual, rng, ns, na), optimizer=ADAM(1e-3)
            ),
            teacher=gpu(agent),
            start_policy=RandomPolicy(Space([-1.0 .. 1.0 for i in 1:na]); rng=rng),
            γ=0.99f0,
            ρ=0.99f0,
            batch_size=256,
            start_steps=0,
            pretraining_steps=pretraining_steps,
            update_freq=1,
            policy_freq=2,
            target_act_limit=1.0,
            target_act_noise=0.1,
            act_limit=1.0,
            act_noise=0.1,
            rng=rng,
            q_bc_weight=q_bc_weight,
            critic_l2_weight=1.0f-6,
            actor_l2_weight=1.0f-6,
            representation_weight=representation_weight,
            similarity_function=similarity_function,
            distill_layer=distill_layer,
        ),
        trajectory=trajectory,
    )

    stop_condition = StopAfterStep(total_steps; is_show_progress=!haskey(ENV, "CI"))
    hook = tensorboard_hook(
        env,
        agent,
        string("logs/", run_name);
        save_checkpoints=true,
        agent_name=string("agents/visual/", run_name),
    )

    return pretrain_run(agent, env, stop_condition, hook)
end

main()
