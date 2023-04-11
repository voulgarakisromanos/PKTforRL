using ReinforcementLearning
using Distances

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

    ε = 1e-7

    teacher_output = transpose(teacher_output)
    student_output = transpose(student_output)

    @assert size(teacher_output)[1] == size(student_output)[1]

    teacher_output_norm = sqrt.(sum(teacher_output.^2,dims=2))
    teacher_output = teacher_output ./ (teacher_output_norm .+ ε)

    student_output_norm = sqrt.(sum(student_output.^2,dims=2))
    student_output = student_output ./ (student_output_norm .+ ε)

    student_similarity = student_output * transpose(student_output)
    teacher_similarity = teacher_output * transpose(teacher_output)

    student_similarity = (student_similarity .+ 1.0) ./ 2.0
    teacher_similarity = (teacher_similarity .+ 1.0) ./ 2.0

    student_similarity = student_similarity./(sum(student_similarity,dims=2) .+ ε)
    teacher_similarity = teacher_similarity./(sum(teacher_similarity,dims=2) .+ ε)

    loss = mean(teacher_similarity .* log.((teacher_similarity .+ ε) ./ (student_similarity .+ ε)))

    return loss
end

function rbf_similarity_loss(student_output::AbstractArray, teacher_output::AbstractArray, gamma)

    teacher_output_norm = sqrt.(sum(teacher_output.^2,dims=1))
    teacher_output = teacher_output ./ (teacher_output_norm .+ eps())

    student_output_norm = sqrt.(sum(student_output.^2,dims=1))
    student_output = student_output ./ (student_output_norm .+ eps())

    # Compute RBF kernel matrices

    student_distances = zeros(size(student_output)[2], size(student_output)[2])
    teacher_distances = zeros(size(teacher_output)[2], size(teacher_output)[2])

    @tullio student_distances[i,j] := (student_output[k,i] - student_output[k,j])^2 
    @tullio teacher_distances[i,j] := (teacher_output[k,i] - teacher_output[k,j])^2 

    K_s = exp.(-gamma * student_distances)
    K_t = exp.(-gamma * teacher_distances)

    # Kernel matrices should add up to 1, as they are interpreted to be probabilities
    K_s  = K_s ./ sum(K_s, dims=2)
    K_t  = K_t ./ sum(K_t, dims=2)

    # Compute loss
    loss = mean(K_t .* log.((K_t .+ eps()) ./ (K_s .+ eps())))

    return loss
end