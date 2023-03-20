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

