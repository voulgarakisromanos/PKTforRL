using Robosuite
using CircularArrayBuffers
using BSON
using Flux
include("../utilities/hooks.jl")
include("../utilities/utils.jl")

image_size = 64
frame_size = 3
env = RoboticEnv(name="Lift", T=Float32, controller="OSC_POSE", enable_visual=true, show=false, horizon=200, image_size=image_size, stop_when_done=true)

na = env.degrees_of_freedom;

BSON.@load "agents/groundtruth/Lift" agent

stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"));

hook = SampleTrajectory{true}(CircularArraySGARTTrajectory(
    capacity = 300_000,
    state = Vector{Float32} => (image_size,image_size,frame_size),
    groundtruth = Vector{Float32} => (size(vcat(vec(env.proprioception_state), vec(env.object_state)))[1]),
    action = Vector{Float32} => (na,)
))

actor_critic_agent = ActorCriticPolicy{false}(agent[:actor], agent[:critic]);

run(actor_critic_agent, env, stop_condition, hook)

BSON.@save "datasets/lift_demo.bson" hook