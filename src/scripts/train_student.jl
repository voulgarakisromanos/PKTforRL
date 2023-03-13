using ReinforcementLearning
using StableRNGs
using Flux
using BSON
using IntervalSets
using TensorBoardLogger
using Logging  
using Robosuite

include("../utilities/CombinedTrajectory.jl")
include("../utilities/hooks.jl")
include("../utilities/utils.jl")
include("../models/network_definitions.jl")
include("../algorithms/TwinDelayedDDPGfromDemos.jl")

image_size = 64;
frame_size = 3;
visual = true;

rng = StableRNG(123);
env = RoboticEnv(name="Lift", T=Float32, controller="OSC_POSE", enable_visual=visual, show=false, horizon=200, image_size=image_size)

na = env.degrees_of_freedom;

init = glorot_uniform(rng)

BSON.@load "datasets/lift_demo.bson" hook

hook = efficient_to_stacked(hook, frame_size=3)

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
        start_steps = 0,
        pretraining_steps = 40_000,
        update_freq = 1,
        policy_freq = 2,
        target_act_limit = 1.0,
        target_act_noise = 0.1,
        act_limit = 1.0,
        act_noise = 0.1,
        rng = rng,
        q_bc_weight = 1.0f0,
        critic_l2_weight = 1.0f-6,
        actor_l2_weight = 1.0f-6,
        representation_weight = 0.0f0,
        logger=TBLogger("log/TD3fromDemonstrations", min_level = Logging.Info)
    ),
    trajectory = trajectory
);

stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"));

hook = TotalRewardPerEpisode()

run(agent, env, stop_condition, hook)

actor = agent.policy.behavior_actor.model |> cpu
critic = agent.policy.behavior_critic.model.critic_nets[1] |> cpu
actor_critic_agent = ActorCriticPolicy{false}(actor, critic);
BSON.@save "agents/visual/lift_visual.bson" actor_critic_agent
