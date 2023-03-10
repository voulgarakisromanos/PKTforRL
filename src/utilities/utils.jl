using ReinforcementLearning

struct ActorCriticPolicy{visual_agent} <: AbstractPolicy
    actor
    critic
end

(policy::ActorCriticPolicy{false})(env) = policy.actor(vcat(vec(env.proprioception_state), vec(env.object_state)))
(policy::ActorCriticPolicy{true})(env) = policy.actor(state(env))


struct CombineActionImageEmbedding{T,F}
    layers::T
    connection::F  #user can pass arbitrary connections here, such as (a,b) -> a + b
end

Flux.@functor CombineActionImageEmbedding

function (combo_layer::CombineActionImageEmbedding)(input)
    (image, action) = input
    combo_layer.connection(combo_layer.layers(image), action)
end