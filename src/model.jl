module Model

using Images
using Reexport

using DSB2018.Data
using DSB2018.Errors

import DSB2018.Errors: metric
import Base: convert, collect, getindex
export getindex

export DSBImage
export convert, metric, grow!, grow

global const Coordinate = CartesianIndex{2}
global const Area{T} = Dict{Coordinate, T}

include("molecular.jl")
@reexport using .Molecular

import .Molecular: start!, next!, done!, belongs, iscell, grow!, getindex

function distance(a::Coordinate, b::Coordinate)::Float64
    δx = a[1] - b[1]
    δy = a[2] - b[2]

    return sqrt(δx^2 + δy^2)
end

struct DSBImage{X}
    data::Data.Image{X}
    cache::Dict{Any, Colony}

    function DSBImage(data::Data.Image{X}) where {X <: Any}
        return new{X}(data, Dict{Any, Colony}())
    end
end

function metric(this::DSBImage{X}, colony::Colony) where {X <: Any}
    truths = collect(values(this.data.masks))
    predictions = [convert(BitArray{2}, x.area, size(this.data.image)) for x in colony.cells]

    return Errors.metric(predictions, truths)
end

function getindex(this::T, x::Float64, y::Float64) where {T <: AbstractMatrix}
    xr = Int64(round(x))
    yr = Int64(round(y))
    return this[xr, yr]
end


function convert(::Type{Area{T}}, m::AbstractMatrix{X}) where {T <: Any, X <: Any}
    out = Area{T}()
    for j in CartesianRange(CartesianIndex{2}(1, 1), CartesianIndex{2}(size(m)))
        out[j] = convert(T, m[j])
    end
    return out
end

function convert(::Type{Area{Bool}}, m::AbstractMatrix{Bool})
    out = Area{Bool}()
    for j in CartesianRange(CartesianIndex{2}(1, 1), CartesianIndex{2}(size(m)))
        !m[j] && continue
        out[j] = true
    end
    return out
end

function convert(::Type{BitArray{2}}, area::Area{T}, sz::Tuple{Int64, Int64})::BitArray{2} where {T <: Any}
    out = BitArray{2}(sz[1], sz[2]) .* false
    for k in keys(area)
        out[k] = true
    end

    return out
end

function convert(::Type{Matrix{X}}, area::Area{T}, sz::Tuple{Int64, Int64})::Matrix{X} where {T <: Any, X <: Any}
    out = zeros(X, sz)
    for (k, v) in area
        out[k] = convert(X, v)
    end

    return out
end

function grow!(this::DSBImage{X}, model::T; show::Bool = false)::Colony where {X <: Any, T <: Any}
    orig_model = deepcopy(model)
    if !haskey(this.cache, orig_model)
        this.cache[orig_model] = grow(this.data.image, model, show = show)
    end

    return this.cache[orig_model]
end


include("models/dmodel.jl")

end
