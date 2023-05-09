using Robosuite
using CircularArrayBuffers
using BSON
using Flux
using TensorBoardLogger
using Logging

include("../utilities/hooks.jl")
include("../utilities/utils.jl")

env_name = "TwoArmPegInHole"

if env_name == "TwoArmPegInHole"
    robots = ("Panda", "Panda")
else
    robots = "Panda"
end

env = RoboticEnv(name=env_name, robots=robots, T=Float32, controller="OSC_POSE", enable_visual=false, show=false, horizon=200, stop_when_done=true)

na = env.degrees_of_freedom;

BSON.@load string("agents/groundtruth/", env_name) agent

stop_condition = StopAfterEpisode(100, is_show_progress=!haskey(ENV, "CI"));

lg = TBLogger(string("logs/",env_name))
total_reward_per_episode = TotalRewardPerEpisode()
total_reward_per_episode.rewards = [0.0]

hook = ComposedHook(SuccessRateHook(success_criterion=()->env.success, logger=lg),
    total_reward_per_episode,
    DoEveryNEpisode() do t, agent, env
        with_logger(lg) do
            @info  "reward" total_reward_per_episode.rewards[end]
        end
    end)

actor_critic_agent = ActorCriticPolicy{false}(agent[:actor], agent[:critic]);

run(actor_critic_agent, env, stop_condition, hook)
