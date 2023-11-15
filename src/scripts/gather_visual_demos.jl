using Robosuite
using CircularArrayBuffers
using BSON
using Flux
include("../utilities/hooks.jl")
include("../utilities/utils.jl")

image_size = 64

env_name = "Door"
robots = "Panda"

env = RoboticEnv(;
    name=env_name,
    robots=robots,
    T=Float32,
    controller="OSC_POSE",
    enable_visual=true,
    show=false,
    horizon=200,
    image_size=image_size,
    stop_when_done=true,
)

na = env.degrees_of_freedom;

BSON.@load string("agents/groundtruth/", env_name) agent

stop_condition = StopAfterStep(10000; is_show_progress=!haskey(ENV, "CI"));

hook = SampleTrajectory{true}(
    CircularArraySGARTTrajectory(;
        capacity=30000,
        state=Vector{Float32} => (image_size, image_size, 3),
        groundtruth=Vector{Float32} =>
            (size(vcat(vec(env.proprioception_state), vec(env.object_state)))[1],),
        action=Vector{Float32} => (na,),
    ),
)

actor_critic_agent = ActorCriticPolicy{false}(agent[:actor], agent[:critic]);

run(actor_critic_agent, env, stop_condition, hook)

dataset = hook.t

BSON.@save string("datasets/", env_name, "_demo.bson") dataset
