module Model

using Images
using Plots
using Juno
using Distributions
using ScikitLearn
using PyCall
using StatsBase
using Interpolations
using SegmentConnection
using DSB2018.Errors
using HypothesisTests
using Distributions

import DSB2018.Errors: metric

@pyimport skimage.filters as skimage_filters
@pyimport skimage.restoration as skimage_restoration
@pyimport skimage.morphology as skimage_morphology
@pyimport scipy.ndimage as ndi

export split_connected_regions, metric

using DSB2018.Data

function construct_densities(images::Vector)
    res = Vector{Matrix{Float64}}(0)
    Juno.@progress "Constructing densities" for img in images
        push!(res, to_density(convert.(Float64, img.image), 3.0, 2, 10000))
    end
    return res
end

function get_scores_by_density(images::Vector{X}, numsamples::Int64) where {X <: Any}

    scores = Vector{Float64}(length(images))
    Juno.@progress "Scoring" for k in eachindex(images)
        raw_image = X == String ? Data.loadTrainImage(images[k]) : images[k]

        truth_images = collect(values(raw_image.masks))
        img = convert.(Float64, raw_image.image)
        res = to_density(img, 1.0, 3, numsamples)

        adaptive_threshold = skimage_filters.threshold_local(res, 21, offset=0.0)
        global_threshold   = skimage_filters.threshold_otsu(res)

        w_adaptve = 0.2
        threshold = adaptive_threshold * w_adaptve .+ global_threshold * (1.0 - w_adaptve)

        img_t = res .> threshold
        img_t = skimage_morphology.remove_small_objects(img_t, min_size=4)
        img_t = ndi.binary_fill_holes(img_t)

        predictions = split_connected_regions(img_t)
        scores[k] = metric(predictions, truth_images)
        m = mean(scores[1:k])
        display(scatter(scores[1:k], title="Mean = $m"))
    end
    return scores
end

function split_connected_regions(this::Union{Matrix{Bool}, BitArray{2}}, x...)::Vector{Matrix{Bool}}
    labeled = Images.label_components(this, x...)
    labels = unique(labeled)
    out = Vector{Matrix{Bool}}(length(labels))
    for k in eachindex(labels)
        new_labels = falses(size(labeled))
        new_labels[labeled .== labels[k]] = true
        out[k] = new_labels
    end

    return out
end

function get_density(this::Vector{Float64}, numsamples::Int64)::Vector{Float64}
    nbins = length(this)
    x = collect(1 : length(this)) .* 1.0
    z = StatsBase.sample(x, Weights(this), numsamples);
    h = StatsBase.fit(Histogram, z, nbins = nbins, closed = :left)

    edges = collect(h.edges[1])[2:end]
    vals  = collect(h.weights)

    vals = vals / sum(vals)
    ρ = interpolate((edges,), vals, Gridded(Linear()))

    ρx = ρ[x]

    zu = unique(z)
    u = Distributions.Uniform(minimum(zu), maximum(zu))
    pval = pvalue(ApproximateOneSampleKSTest(zu, u))

    return ρx / sum(ρx) / length(ρx) * nbins * pval
end

function distance(x1::CartesianIndex, x2::CartesianIndex)::Float64
    return sqrt((x2[1] - x1[1])^2 + (x2[2] - x1[2])^2)
end

function box_mean(this::Vector{Float64}, n::Int64)::Vector{Float64}
    out = similar(this)
    for k in eachindex(this)
        j1 = max(1, k - n)
        j2 = min(length(this), k + n)
        v = 0.0
        nn = 0.0
        for j in j1 : j2
            v += this[j]
            nn += 1
        end
        out[k] = v / nn
    end
    return out
end

function to_density(this::Matrix{Float64},
                    σ::Float64,
                    nlag::Int64,
                    numsamples::Int64)

    this = skimage_restoration.denoise_bilateral(this, multichannel=false)

    out = zeros(size(this))
    num_visits = zeros(size(this))

    nr = size(this, 1)
    nc = size(this, 2)

    qs = mean(this[:])
    if qs > 0.5
        this = 1.0 - this
    end

    for k = 1 : size(this, 1)
        mvalues = this[k, :]
        densities = get_density(mvalues, numsamples)
        if nlag != 1
            densities *= StatsBase.autocor(densities, [nlag])[1]
        end

        out[k, :] .+= densities
        num_visits[k, :] .+= 1
    end

    for k = 1 : size(this, 2)
        mvalues = this[:, k]
        densities = get_density(mvalues, numsamples)
        if nlag != 1
            densities *= StatsBase.autocor(densities, [nlag])[1]
        end
        out[:, k] .+= densities
        num_visits[:, k] .+= 1
    end

    res = out ./ num_visits# .* uniform_weights / 2.
    return res
end

end
