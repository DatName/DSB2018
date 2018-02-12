module Model

using Distances
using DSB2018.Data
using DSB2018.Errors

import Base: isless, ∈
export isless, ∈, Cell
export start, next, done, collect

using Images
import Base: start, next, done, collect
import ImageView: imshow
import Base: convert
import DSB2018.Errors: metric

export convert, imshow, collect, metrics

const Coordinate = CartesianIndex{2}
const Area = Dict{Coordinate, Float64}

∈(x::Coordinate, a::Area) = haskey(a, x)

function neighbors(this::Coordinate, numrows::Int64, numcols::Int64; r::Int64 = 1)::Vector{Coordinate}
    res = Coordinate[]

    h = -r:1:r
    v = -r:1:r
    for ik in h, ij in v
        if ik == 0 && ij == 0
            continue
        end

        x, y = this[1] + ik, this[2] + ij
        if x > 0 && y > 0
            if x > numrows
                continue
            end

            if y > numcols
                continue
            end

            push!(res, CartesianIndex{2}(x, y))
        end
    end

    return res
end

function distance(x::Coordinate)::Float64
    return sqrt(x[1]^2 + x[2]^2)
end

function distance(x::Coordinate, y::Coordinate)::Float64
    return distance(x - y)
end

include("./shapes/line.jl")
include("./shapes/set.jl")

function distance_no_nearest(area::Vector{Coordinate})::Vector{Float64}
    out = Vector{Float64}(length(area))
    for k in eachindex(area)
        base = area[k]
        m = Inf
        for x in area
            if x == area[k]
                continue
            end

            δ = distance(base, x)
            if δ < m
                m = δ
            end
        end

        out[k] = m
    end

    return out
end

function getImages()
    out = Data.Image[]
    for (k, v) in Data.TrainImages
        push!(out, v)
    end
    return out
end

struct ImagePrediction
    image::Data.Image
    prediction::Dict{UInt64, ThickSet}

    function ImagePrediction(image::Data.Image)
        return new(image, Dict{UInt64, ThickSet}())
    end
end

function collect(this::ImagePrediction,
                    avgsize::Int64,
                    qsharpness::Float64,
                    kernel::T) where {T <: AbstractArray{Float64, 2}}

    img = imfilter(this.image.image, kernel)
    sharpness = quantile(img[:], qsharpness)
    @show sharpness

    s = ThickSet(img, avgsize, sharpness)
    for v in values(this.prediction)
        merge!(s.covered, v.covered)
    end

    collect(s)

    img_2_b = convert(Matrix{Bool}, s);
    img_2_r = convert(Matrix{Float64}, img_2_b)
    img_2_k = imfilter(img_2_r, Kernel.gaussian(0.2))

    s_h = ThickSet(img_2_k, 2, sharpness/2.)
    collect(s_h)

    if !isempty(s_h.lines)
        h = hash(avgsize)
        h = hash(qsharpness, h)
        h = hash(collect(kernel), h)

        this.prediction[h] = s_h
    end

    return s_h
end

function get_truth_images(this::ImagePrediction)
    return collect(values(this.image.masks))
end

function get_prediction_images(this::ThickSet)
    prediction_images = Matrix{Bool}[]

    for l in this.lines
        x = convert(Matrix{Bool}, l)
        push!(prediction_images, x)
    end

    return prediction_images
end

function metrics(this::ImagePrediction)::Dict{UInt64, Float64}
    out = Dict{UInt64, Float64}()
    for (k, v) in this.prediction
        out[k] = metric(this, v)
    end

    return out
end

function metric(this::ImagePrediction, that::ThickSet)::Float64
    truth_images = get_truth_images(this)
    prediction_images = get_prediction_images(that)

    return Errors.metric(prediction_images, truth_images)
end


end
