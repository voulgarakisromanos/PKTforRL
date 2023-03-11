using StatsBase
using ReinforcementLearning
using Random

function combine_named_tuples(tuple1, tuple2)
    @assert keys(tuple1)==keys(tuple2)
    tuple_keys = keys(tuple1)
    combined_tuple = NamedTuple(key => (typeof(tuple1[key]) <: Vector ? vcat(tuple1[key],tuple2[key]) : cat(tuple1[key],tuple2[key], dims=4)) for key in tuple_keys)
end

struct CombinedTrajectory{T} <: AbstractTrajectory
    main_trajectory::T
    demo_trajectory::T
    ratio::AbstractFloat
end

Base.keys(t::CombinedTrajectory) = keys(t.main_trajectory)
Base.length(t::CombinedTrajectory) = length(t.main_trajectory)
Base.getindex(t::CombinedTrajectory, s::Symbol) = t.main_trajectory[s]

function (s::BatchSampler)(t::CombinedTrajectory)
    demo_sample_length = Int(round(t.ratio * s.batch_size))
    main_sample_length = s.batch_size - demo_sample_length
    s.cache = nothing
    custom_s = deepcopy(s)
    custom_s.batch_size = main_sample_length
    _, main_samples = sample(s.rng, t.main_trajectory, custom_s)
    custom_s.cache = nothing
    custom_s.batch_size = demo_sample_length
    _, demo_samples = sample(s.rng, t.demo_trajectory, custom_s)
    return nothing, combine_named_tuples(main_samples, demo_samples)
end

function StatsBase.sample(rng::AbstractRNG, t::CombinedTrajectory, s::NStepBatchSampler)
    demo_sample_length = Int(round(t.ratio * s.batch_size))
    main_sample_length = s.batch_size - demo_sample_length
    s.cache = nothing
    custom_s = deepcopy(s)
    custom_s.batch_size = main_sample_length
    _, main_samples = sample(s.rng, t.main_trajectory, custom_s)
    custom_s.cache = nothing
    custom_s.batch_size = demo_sample_length
    _, demo_samples = sample(s.rng, t.demo_trajectory, custom_s)
    return nothing, combine_named_tuples(main_samples, demo_samples)
end