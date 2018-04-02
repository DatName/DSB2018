module Resampling

using Juno
using PyCall
using StatsBase
using DSB2018
using DSB2018.Data
using DSB2018.Model
using ColorTypes
using Images

import StatsBase: predict
@pyimport skimage.restoration as skimage_restoration
@pyimport skimage.morphology as skimage_morphology
@pyimport scipy.ndimage as ndi
@pyimport skimage.filters as skimage_filters

function re_sample(img::Matrix{Float64}, n::Int64)
    jx = collect(1 : length(img))
    w = StatsBase.ProbabilityWeights(img[:])

    out = zeros(length(jx))
    Juno.@progress "Sampling" for k = 1 : n
        x = sample(jx, w, (length(jx), ))
        out[x] .+= 1.
    end

    reshape(out, size(img))/n
end

function re_sample(img::DSB2018.Data.Image, n::Int64)
    return re_sample(Float64.(Gray.(img.image)), n)
end

function is_bi_modal(x::X) where {X <: AbstractArray}
    dx = diff(x)
    return sum(sign.(dx[1:(end-1)]) .!= sign.(dx[2:end])) == 3
end

function bi_modal_picks(x::X) where {X <: AbstractArray}
    pick1 = findfirst(diff(x) .< 0.0)
    pick2 = findlast(diff(x) .> 0.0)
    return pick1, pick2
end


end
