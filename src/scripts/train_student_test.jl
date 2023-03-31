include("../utilities/training_dependencies.jl")

env = "Lift"
pretraining_steps = 0
representation_weight = 0.05f0
q_bc_weight = 2.5f0
run_name = "test_run"

image_size = 64;
frame_size = 3;
visual = true;

rng = StableRNG(123);
env = RoboticEnv(name=env, T=Float32, controller="OSC_POSE", enable_visual=visual, show=false, horizon=200, image_size=image_size)

na = env.degrees_of_freedom;
ns = size(vcat(vec(env.proprioception_state), vec(env.object_state)))[1]

BSON.@load "datasets/lift_demo.bson" dataset
BSON.@load "agents/groundtruth/Lift" agent
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
        ),
    trajectory = trajectory
);

stop_condition = StopAfterStep(10, is_show_progress=!haskey(ENV, "CI"));
hook = tensorboard_hook(agent, string("logs/",run_name), save_checkpoints=true, agent_name=string("agents/visual/",run_name))

pretrain_run(agent, env, stop_condition, hook)
