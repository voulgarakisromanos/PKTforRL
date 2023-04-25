include("../utilities/training_dependencies.jl")


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--env"
            help = "Environment name, e.g Lift or Door"
            arg_type = String
            default = "Lift"
        "--run_name"
            help = "Run name for logging purposes"
            arg_type = String
            required = true
        "--pre_steps"
            help = "Number of pretraining steps"
            arg_type = Int
            default = 20_000
        "--repr_weight"
            help = "Representation loss weight"
            arg_type = Float32
            default = 1.0f0
        "--lambda"
            help = "Relative weight for Q and BC losses"
            arg_type = Float32
            default = 2.5f0
        "--gamma"
            help = "Width for RBF-like similarity kernels"
            arg_type = Float32
            default = 1.0f0
        "--similarity_function"
            help = "RBF or cosine"
            arg_type = String
            default = "cosine"
    end

    return parse_args(s)
end


function main()

    parsed_args = parse_commandline()
    env = parsed_args["env"]
    pretraining_steps = parsed_args["pre_steps"]
    representation_weight = parsed_args["repr_weight"]
    q_bc_weight = parsed_args["lambda"]
    run_name = parsed_args["run_name"]
    kernel_width = parsed_args["gamma"]
    similarity_function_name = parsed_args["similarity_function"]

    if similarity_function_name == "RBF"
        similarity_function = (x, y) -> rbf_similarity_loss(x, y, kernel_width)
    elseif similarity_function_name == "cosine"
        similarity_function = (x, y) -> cosine_similarity_loss(x, y)
    elseif similarity_function_name == "linear"
        similarity_function = (x, y) -> linear_similarity_loss(x, y)
    else
        println("No similarity loss selected")
    end


    image_size = 64;
    frame_size = 3;
    visual = true;

    rng = StableRNG(123);
    env = RoboticEnv(name=env, T=Float32, controller="OSC_POSE", enable_visual=visual, show=false, horizon=200, image_size=image_size)

    na = env.degrees_of_freedom;
    ns = size(vcat(vec(env.proprioception_state), vec(env.object_state)))[1]

    BSON.@load string("datasets/", env, "_demo.bson") dataset
    BSON.@load string("agents/groundtruth/", env) agent
    teacher = agent

    demo_trajectory = dataset

    trajectory = CombinedTrajectory(CircularArraySGARTTrajectory(
        capacity = 300_000,
        state = Vector{Float32} => (image_size, image_size, frame_size),
        groundtruth = Vector{Float32} => (size(vcat(vec(env.proprioception_state), vec(env.object_state)))[1]),
        action = Vector{Float32} => (na,)
    ), demo_trajectory, 0.25)

    agent = Agent(
        policy = TwinDelayedDDPGPolicy(
            behavior_actor = NeuralNetworkApproximator(
                model = create_actor(visual, rng, ns, na),
                optimizer = ADAM(1e-4),
            ),
            behavior_critic = NeuralNetworkApproximator(
                model = create_critic(visual, rng, ns, na),
                optimizer = ADAM(1e-4),
            ),
            target_actor = NeuralNetworkApproximator(
                model = create_actor(visual, rng, ns, na),
                optimizer = ADAM(1e-4),
            ),
            target_critic = NeuralNetworkApproximator(
                model = create_critic(visual, rng, ns, na),
                optimizer = ADAM(1e-4),
            ),
            teacher = teacher |> gpu,
            start_policy = RandomPolicy(Space([-1.0..1.0 for i=1:na]); rng = rng),
            γ = 0.99f0,
            ρ = 0.99f0,
            batch_size = 256,
            start_steps = 0,
            pretraining_steps = pretraining_steps,
            update_freq = 1,
            policy_freq = 2,
            target_act_limit = 1.0,
            target_act_noise = 0.1,
            act_limit = 1.0,
            act_noise = 0.1,
            rng = rng,
            q_bc_weight = q_bc_weight,
            critic_l2_weight = 1.0f-6,
            actor_l2_weight = 1.0f-6,
            representation_weight = representation_weight,
            similarity_function = similarity_function
            ),
        trajectory = trajectory
    );

    stop_condition = StopAfterStep(100_000, is_show_progress=!haskey(ENV, "CI"));
    hook = tensorboard_hook(agent, string("new_logs/",run_name), save_checkpoints=true, agent_name=string("agents/visual/",run_name))

    pretrain_run(agent, env, stop_condition, hook)
end

main()