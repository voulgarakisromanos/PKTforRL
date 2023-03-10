using ReinforcementLearning

mutable struct SampleTrajectory <: AbstractHook
    t::AbstractTrajectory
end

function (hook::SampleTrajectory)(::PreActStage, agent, env, action)
    push!(hook.t, state=state(env), action=action)
end

function (hook::SampleTrajectory)(::PostActStage, agent, env)
    push!(hook.t, reward=reward(env), terminal=is_terminated(env))
end

mutable struct EfficientFramesHook <: AbstractHook
    t::AbstractTrajectory
end

function (hook::EfficientFramesHook)(::PreActStage, agent, env, action)
    if isempty(hook.t[:state])
        for i = 1:size(env.image_buffer)[end]
            push!(hook.t, state=env.image_buffer[:, :, i])
        end
    else
        push!(hook.t, state=env.image_buffer[:, :, end])
    end
    push!(hook.t, action=action[1])
end

function (hook::EfficientFramesHook)(::PostActStage, agent, env)
    push!(hook.t, reward=reward(env), terminal=is_terminated(env))
    # if reward(env) == 0  # keep only important transitions
    #     pop!(hook.t) 
    # end
end


function efficient_to_stacked(hook::EfficientFramesHook; frame_size=4)
    x_size, y_size, ultimate_pointer = size(hook.t[:state])
    stacked_hook = StateImageTransition(CircularArraySARTTrajectory(
        capacity=30000,
        state=Vector{Float32} => (x_size, y_size, frame_size),
        action=typeof(hook.t[:action][1]) => (size(hook.t[:action])[1],)
    ))
    start_pointer = 1
    end_pointer = frame_size
    while end_pointer <= ultimate_pointer
        frame_array = StackFrames(Float32, x_size, y_size, frame_size)
        for frame_index = start_pointer:end_pointer
            frame_array(hook.t[:state][:, :, frame_index])
        end
        push!(stacked_hook.t, state=frame_array)
        start_pointer += 1
        end_pointer += 1
    end
    for i = 1:length(hook.t)
        push!(stacked_hook.t, action=hook.t[:action][i], reward=hook.t[:reward][i], terminal=hook.t[:terminal][i])
    end
    return stacked_hook
end

mutable struct StateImageTransition <: AbstractHook
    t::AbstractTrajectory
end

function (hook::StateImageTransition)(::PreActStage, agent, env, action)
    push!(hook.t, state=env.grayscale, action=action[1])
end

function (hook::StateImageTransition)(::PostActStage, agent, env)
    push!(hook.t, reward=reward(env), terminal=is_terminated(env))
end