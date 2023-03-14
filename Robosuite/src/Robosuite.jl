# Robosuite.jl
module Robosuite

export RoboticEnv

using ReinforcementLearning
using PythonCall
using CircularArrayBuffers
using IntervalSets
using Images: Gray, colorview, RGB

mutable struct RoboticEnv{T<:AbstractFloat,enable_visual} <: AbstractEnv
    ptr::Py
    visual::Bool
    show::Bool
    degrees_of_freedom::Int
    image_buffer::StackFrames{T}
    proprioception_state::Vector{T}
    object_state::Vector{T}
    reward::T
    done::Bool
    frame_stack_size::Int
    timestep::Int
    horizon::Int
end

function RoboticEnv(; name="Lift", robots="Panda", controller="OSC_POSE", show=false, T=Float32, enable_visual=false, horizon=500, image_size=84, frame_stack_size=3, control_freq=20)
    suite = pyimport("robosuite")
    config = suite.load_controller_config(default_controller=controller)
    ptr = suite.make(
        env_name=name, # try with other tasks like "Stack" and "Door"
        robots=robots,  # try with other robots like "Sawyer" and "Jaco"
        render_camera="agentview",
        has_renderer=show,
        has_offscreen_renderer=enable_visual,
        use_camera_obs=enable_visual,
        reward_shaping=true,
        ignore_done=true,
        control_freq=control_freq,
        camera_heights=image_size,
        camera_widths=image_size,
        horizon=horizon,
        hard_reset=false,
        reward_scale=1.0,
        controller_configs=config
    )
    degrees_of_freedom = pyconvert(Int, ptr.action_dim)
    image_buffer = StackFrames(T, image_size, image_size, frame_stack_size)
    obs, _, _, _ = ptr.step(rand(degrees_of_freedom))
    proprioception_size = size(pyconvert(Vector, obs["robot0_proprio-state"]))[1]    
    proprioception_state = Vector{T}(undef, proprioception_size)
    object_size = size(pyconvert(Vector, obs["object-state"]))[1]    
    object_state = Vector{T}(undef, object_size)
    reward = T(0)
    done = false
    return RoboticEnv{T,enable_visual}(ptr, enable_visual, show, degrees_of_freedom, image_buffer, proprioception_state, object_state, reward, done, frame_stack_size, 0, horizon)
end

function common_env_procedure(env::RoboticEnv{T}, action::Vector) where {T}
    obs, reward, done, info = env.ptr.step(action)
    env.reward = pyconvert(T, reward)
    env.done = pyconvert(Bool, done)
    if env.timestep > env.horizon || env.reward < 1e-5
        env.done = true
    # elseif env.reward == 1.0
    #     env.done = true
        # env.reward = env.horizon - env.timestep
    end
    env.proprioception_state = pyconvert(Vector{T}, obs["robot0_proprio-state"])
    env.object_state = pyconvert(Vector{T}, obs["object-state"])
    if env.show
        env.ptr.render()
    end
    env.timestep += 1
    return obs
end

(env::RoboticEnv{T,false})(action::Vector) where {T<:Number} = common_env_procedure(env, action)

function (env::RoboticEnv{T,true})(action::Vector) where {T<:Number} 
    obs = common_env_procedure(env, action)
    env.image_buffer(convert_ndarray_to_grayscale(obs["agentview_image"])) 
end

function convert_ndarray_to_grayscale(obs)
    x = pyconvert(Array, obs)./255.0f0
    x = permutedims(x, [3,1,2])
    x = colorview(RGB, x)
    x = Gray.(x)
end

function RLBase.state_space(env::RoboticEnv{T}) where {T<:Number}
    return Space([-1.0 .. 1.0 for i = 1:(size(env.proprioception_state)[1]+size(env.object_state)[1])])
end

RLBase.action_space(env::RoboticEnv) = Space([-1.0 .. 1.0 for i = 1:env.degrees_of_freedom])
RLBase.reward(env::RoboticEnv) = env.reward
RLBase.is_terminated(env::RoboticEnv) = env.done

function RLBase.state(env::RoboticEnv{T,false}) where {T<:Number}
    return vcat(vec(env.proprioception_state), vec(env.object_state))
end

function RLBase.state(env::RoboticEnv{T,true}) where {T<:Number}
    # return (env.image_buffer,env.proprioception_state)
    return env.image_buffer
end

function RLBase.reset!(env::RoboticEnv{T}) where {T<:Number}
    env.ptr.reset()
    env.reward = typeof(env.reward)(0)
    env.done = false
    env.timestep = 0
    for i=1:env.frame_stack_size
        env(T.(zeros(env.degrees_of_freedom)))
    end
end

Base.close(env::RoboticEnv) = env.ptr.close()
Base.show(env::RoboticEnv) = env.ptr.render()

end