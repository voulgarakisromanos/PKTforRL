using StatsBase
using ReinforcementLearning
using Random

function combine_named_tuples(tuple1, tuple2)
    @assert keys(tuple1) == keys(tuple2)
    tuple_keys = keys(tuple1)
    return combined_tuple = NamedTuple(
        key => (cat(tuple1[key], tuple2[key]; dims=ndims(tuple1[key]))) for
        key in tuple_keys
    )
end

struct CombinedTrajectory{T} <: AbstractTrajectory
    main_trajectory::T
    demo_trajectory::T
    ratio::AbstractFloat
end

Base.keys(t::CombinedTrajectory) = keys(t.main_trajectory)
Base.length(t::CombinedTrajectory) = length(t.main_trajectory)
Base.getindex(t::CombinedTrajectory, s::Symbol) = t.main_trajectory[s]
