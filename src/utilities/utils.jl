using ReinforcementLearning

struct ActorCriticPolicy{visual_agent} <: AbstractPolicy
    actor
    critic
end

(policy::ActorCriticPolicy{false})(env) = policy.actor(vcat(vec(env.proprioception_state), vec(env.object_state)))
(policy::ActorCriticPolicy{true})(env) = vec(policy.actor(Flux.unsqueeze(state(env), dims=4)))

struct CombineActionImageEmbedding{T,F}
    layers::T
    connection::F  #user can pass arbitrary connections here, such as (a,b) -> a + b
end

Flux.@functor CombineActionImageEmbedding

function (combo_layer::CombineActionImageEmbedding)(input)
    (image, action) = input
    combo_layer.connection(combo_layer.layers(image), action)
end

function normalise_difference_image(img)
    img = img .- minimum(img)
    img = img ./ maximum(img)
end

function save_agent(policy_agent, name, visual=true)
    actor = policy_agent.policy.behavior_actor.model |> cpu
    critic = policy_agent.policy.behavior_critic.model.critic_nets[1] |> cpu
    agent = ActorCriticPolicy{visual}(actor, critic);
    BSON.@save name agent;
end

function cosine_similarity_loss(student_output::AbstractArray, teacher_output::AbstractArray)

    teacher_output = transpose(teacher_output)
    student_output = transpose(student_output)

    @assert size(teacher_output)[1] == size(student_output)[1]

    teacher_output_norm = sqrt.(sum(teacher_output.^2,dims=2))
    teacher_output = teacher_output ./ teacher_output_norm

    student_output_norm = sqrt.(sum(student_output.^2,dims=2))
    student_output = student_output ./ student_output_norm

    student_similarity = student_output * transpose(student_output)
    teacher_similarity = teacher_output * transpose(teacher_output)

    student_similarity = (student_similarity .+ 1.0) ./ 2.0
    teacher_similarity = (teacher_similarity .+ 1.0) ./ 2.0

    student_similarity = student_similarity./sum(student_similarity,dims=2)
    teacher_similarity = teacher_similarity./sum(teacher_similarity,dims=2)

    ϵ = 1e-7

    loss = mean(teacher_similarity .* log.((teacher_similarity .+ ϵ) ./ (student_similarity .+ ϵ)))

    return loss
end