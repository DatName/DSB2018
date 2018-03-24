module Model

using Flux
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
using ImageTransformations
using CoordinateTransformations

import DSB2018.Errors: metric

@pyimport skimage.filters as skimage_filters
@pyimport skimage.restoration as skimage_restoration
@pyimport skimage.morphology as skimage_morphology
@pyimport scipy.ndimage as ndi

export split_connected_regions, metric

using DSB2018.Data

function test_metric(id::String)
    img = Data.loadTrainImage(id)
    this = convert.(Float64, convert.(Gray{Normed{UInt8,8}}, img.image))

    if mean(this) > 0.5
        this = 1.0 - this
    end

    tfm  = recenter(RotMatrix(pi/600), ImageTransformations.center(this));
    this = warp(this, tfm)
    this = collect(this)
    this[isnan.(this)] = 0.0

    if size(this, 1) > size(img.mask, 1)
        this = this[1:size(img.mask, 1), :]
    end
    while size(this, 1) < size(img.mask, 1)
        this = [this; zeros(1, size(this, 2))]
    end

    if size(this, 2) > size(img.mask, 2)
        this = this[:, 1:size(img.mask, 2)]
    end
    while size(this, 2) < size(img.mask, 2)
        this = [this zeros(size(this, 1), 1)]
    end

    @assert all(size(this) .== size(img.mask))

    th = skimage_filters.threshold_otsu(this)

    labeled_prediction = Images.label_components(this .> th)
    labeled_prediction[labeled_prediction .== 1] = 0

    prediction_vector  = split_labeled_matrix(labeled_prediction)
    truth_images = getTrainImageMasks(img)

    mask = img.mask
    Images.save("/home/korelin/mask.png", (mask .> 0)*1.0)
    Images.save("/home/korelin/image.png", (this .> th)*1.0)

    pt = Images.label_components((this .> th)*1.0)
    mt = Images.label_components(mask .> 0)

    truth_mt  = split_labeled_matrix(mt)
    pred_pt   = split_labeled_matrix(pt)

    return metric(pred_pt, truth_mt)
end

function get_predictions_by_density(images::Vector)::Vector{Prediction}
    predictions = Vector{Prediction}(length(images))
    t = Vector{Float64}(0)
    num_labels = Vector{Int64}(0)

    Juno.@progress "Scoring" for k in eachindex(images)
        tic()
        X = typeof(images[k])
        raw_image = X == String ? loadTrainImage(images[k]) : images[k]

        res = to_density(raw_image.image, 4)

        if any(isinf.(res) .| isnan.(res))
            throw("At image $k: non-finite or nan res")
        end

        adaptive_threshold = skimage_filters.threshold_local(res, 41, offset=0.0)
        global_threshold   = skimage_filters.threshold_otsu(res)

        w_adaptve = 0.5
        threshold = adaptive_threshold * w_adaptve .+ global_threshold * (1.0 - w_adaptve)

        img_t = res .> threshold
        img_t = skimage_morphology.remove_small_objects(img_t, min_size=4)
        img_t = ndi.binary_fill_holes(img_t)

        labeled_prediction = Images.label_components(img_t)
        prediction_vector  = split_labeled_matrix(labeled_prediction)

        truth_images = getTrainImageMasks(raw_image)
        if !isempty(truth_images)
            score = metric(prediction_vector, truth_images)
        else
            score = NaN
        end

        predictions[k] = Prediction(raw_image.id, score, labeled_prediction)

        t0 = toq()

        push!(t, t0)
        push!(num_labels, length(prediction_vector))
        n = sum(t)

        scores = [x.score for x in predictions[1:k]]
        m = mean(scores)

        p1 = scatter(scores, title="Mean = $m", legend = false)
        p2 = plot(t, fill = (0, .9, :grey), title="Seconds. Total: $n", legend = false)

        if length(num_labels) < 2
            #some sort of a bug: bar([1]) errors
            display(plot(p1, p2, layout = grid(2, 1)))
            continue
        end

        p3 = bar(num_labels,  fillcolor=:grey, title="Num labels", legend = false)
        display(plot(p1, p2, p3, layout = grid(3, 1)))
    end

    return predictions
end

function split_labeled_matrix(this::Matrix{Int64})::Vector{Matrix{Bool}}
    labels = unique(this)
    labels = labels[labels .> 0]

    out = Vector{Matrix{Bool}}(length(labels))
    for k in eachindex(labels)
        new_labels = falses(size(this))
        new_labels[this .== labels[k]] = true
        out[k] = new_labels
    end

    return out
end

function get_density(this::Vector{Float64})::Vector{Float64}
    st = sum(this)
    if st .< 1e-3
        return zeros(size(this))
    end

    return this / sum(this) / length(this) * 100
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

function to_density(colored_input::Matrix{X}, nlag::Int64) where {X <: Any}

    if X <: ColorTypes.AbstractGray
        colored_input = convert.(Gray{Normed{UInt8,8}}, colored_input)
    end

    this = convert.(Float64, colored_input)
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
        densities = get_density(mvalues)
        if (nlag != 1) && (sum(densities) > 0.0)
            densities *= StatsBase.autocor(densities, [nlag])[1]
        end

        out[k, :] .+= densities
        num_visits[k, :] .+= 1
    end

    for k = 1 : size(this, 2)
        mvalues = this[:, k]
        densities = get_density(mvalues)
        if (nlag != 1) && (sum(densities) > 0.0)
            densities *= StatsBase.autocor(densities, [nlag])[1]
        end
        out[:, k] .+= densities
        num_visits[:, k] .+= 1
    end

    res = out ./ num_visits# .* uniform_weights / 2.
    return res
end

end
