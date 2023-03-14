using ReinforcementLearning
using StableRNGs
using Robosuite
using Flux
using BSON
using IntervalSets
using TensorBoardLogger
using Logging 
using CircularArrayBuffers

include("../utilities/CombinedTrajectory.jl")
include("../utilities/hooks.jl")
include("../utilities/utils.jl")
include("../models/network_definitions.jl")
include("../algorithms/TwinDelayedDDPGfromDemos.jl")

image_size = 64;
frame_size = 3;
visual = true;
env = RoboticEnv(name="Lift", T=Float32, controller="OSC_POSE", enable_visual=visual, show=false, horizon=200, image_size=image_size)

BSON.@load "agents/groundtruth/Lift" agent

stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"));

hook = SampleTrajectory(CircularArraySARTTrajectory(
    capacity = 30000,
    state = Vector{Float32} => (image_size,image_size,3),
    action = Vector{Float32} => (7,)
))

actor_critic_agent = ActorCriticPolicy{false}(agent[:actor], agent[:critic]);

run(actor_critic_agent, env, stop_condition, hook)

rng = StableRNG(123);
na = env.degrees_of_freedom;
init = glorot_uniform(rng)

demo_trajectory = hook.t

trajectory = CombinedTrajectory(CircularArraySARTTrajectory(
    capacity = 300_000,
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
        pretraining_steps = 80_000,
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
        ),
    trajectory = trajectory
);

stop_condition = StopAfterStep(300_000, is_show_progress=!haskey(ENV, "CI"));
hook = tensorboard_hook(agent, "logs/Lift")

run(agent, env, stop_condition, hook)

actor = agent.policy.behavior_actor.model |> cpu
critic = agent.policy.behavior_critic.model.critic_nets[1] |> cpu
visual_agent = ActorCriticPolicy{false}(actor, critic);
BSON.@save "agents/visual/lift_visual.bson" visual_agent