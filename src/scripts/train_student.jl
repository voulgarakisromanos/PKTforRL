using ReinforcementLearning
using StableRNGs
using Flux
using BSON
using IntervalSets
using TensorBoardLogger
using Logging  
using Robosuite

include("../utilities/CombinedTrajectory.jl")
include("../models/network_definitions.jl")
include("../../src/utilities/hooks.jl")
include("../algorithms/TwinDelayedDDPGfromDemos.jl")

image_size = 64;
frame_size = 3;
visual = true;

rng = StableRNG(123);
env = RoboticEnv(name="Lift", T=Float32, controller="OSC_POSE", enable_visual=visual, show=true, horizon=200, image_size=image_size)

na = env.degrees_of_freedom;
ns = size(env.proprioception_state)[1]+size(env.object_state)[1]

init = glorot_uniform(rng)

demo_trajectory = hook.t

trajectory = CombinedTrajectory(CircularArraySARTTrajectory(
    capacity = 100_000,
    state = Vector{Float32} => (image_size,image_size, frame_size),
    action = Vector{Float32} => (na,),
), demo_trajectory, 0.25)

agent = Agent(
    policy = TwinDelayedDDPGPolicy(
        behavior_actor = NeuralNetworkApproximator(
            model = create_actor(visual),
            optimizer = ADAM(1e-4),
        ),
        behavior_critic = NeuralNetworkApproximator(
            model = create_critic(visual),
            optimizer = ADAM(1e-4),
        ),
        target_actor = NeuralNetworkApproximator(
            model = create_actor(visual),
            optimizer = ADAM(1e-4),
        ),
        target_critic = NeuralNetworkApproximator(
            model = create_critic(visual),
            optimizer = ADAM(1e-4),
        ),
        start_policy = RandomPolicy(Space([-1.0..1.0 for i=1:na]); rng = rng),
        γ = 0.99f0,
        ρ = 0.99f0,
        batch_size = 256,
        start_steps = 5500,
        pretraining_steps = 10,
        update_freq = 1,
        policy_freq = 2,
        target_act_limit = 1.0,
        target_act_noise = 0.1,
        act_limit = 1.0,
        act_noise = 0.1,
        rng = rng,
        q_bc_weight = 1.0f0,
        critic_l2_weight = 1.0f-5,
        actor_l2_weight = 1.0f-5,
        representation_weight = 0.0f0,
        logger=TBLogger("log/TD3fromDemonstrations", min_level = Logging.Info)
    ),
    trajectory = trajectory
);

stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"));

hook = TotalRewardPerEpisode()

# pretrain(agent, 10)

run(agent, env, stop_condition, hook)