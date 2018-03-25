__precompile__(false)

module Errors

using MLBase
using StatsBase

Coordinate = CartesianIndex{2}
const Area = Dict{Coordinate, Bool}

export metric

function intersection_over_union(this::Matrix{Bool}, that::Matrix{Bool})::Float64
    ni = sum(this .& that)
    nu = sum(this .| that)
    return ni / nu
end

function intersection_over_union(this::Area, that::Area)::Float64
    ni = 0

    for x in keys(this)
        if !haskey(that, x)
            continue
        end
        ni += 1
    end

    if ni == 0
        return 0.0
    end

    nu = length(merge(this, that))

    return ni / nu
end

function non_zero_area(this::T)::Area where {T <: AbstractMatrix}
    linear_indices = find(this)
    out = Area()
    for k in eachindex(linear_indices)
        r, c = ind2sub(size(this), linear_indices[k])
        out[CartesianIndex{2}(r, c)] = true
    end

    return out
end

function assert_empty_intersection(this::Vector{T})::Void where {T <: AbstractMatrix}
    covered_area = Area()
    for img in this
        area = non_zero_area(img)
        @assert isempty(intersect(collect(keys(area)), collect(keys(covered_area))))
        merge!(covered_area, area)
    end

    return nothing
end

function convert_to_binary(prediction_images::Vector{T},
                            truth_images::Vector{G},
                            thresholds::Z)::Dict{Float64, Tuple{Vector{Bool}, Vector{Bool}} } where {T <: AbstractMatrix, G <: AbstractMatrix, Z <: AbstractArray}

    base_truth_image_areas = Dict{Int64, Area}()
    [base_truth_image_areas[x] = non_zero_area(truth_images[x]) for x in eachindex(truth_images)]

    ious = Matrix{Float64}(length(prediction_images), length(base_truth_image_areas))
    for k in eachindex(prediction_images)
        img_area = non_zero_area(prediction_images[k])
        for (idx, truth_area) in base_truth_image_areas
            ious[k, idx] = intersection_over_union(img_area, truth_area)
        end
    end

    predictions_per_threshold = Dict{Float64, Tuple{Vector{Bool}, Vector{Bool}}}()

    for τidx in eachindex(thresholds)
        threshold_predictions = Vector{Bool}(length(prediction_images))
        threshold_labels      = Vector{Bool}(length(prediction_images))

        τ = thresholds[τidx]
        truth_image_areas = copy(base_truth_image_areas)

        for k in eachindex(prediction_images)
            img = prediction_images[k]
            score = 0.0

            for (idx, truth_area) in truth_image_areas
                score = ious[k, idx]

                if score > τ
                    delete!(truth_image_areas, idx)
                    break
                end
            end

            threshold_predictions[k] = true
            threshold_labels[k] = score > τ
        end

        for idx in truth_image_areas
            push!(threshold_predictions, false)
            push!(threshold_labels, true)
        end

        predictions_per_threshold[τ] = (threshold_labels, threshold_predictions)
    end

    return predictions_per_threshold
end

"Result for a single image"
function metric(prediction_images::Vector{T},
                truth_images::Vector{G},
                τ::Z)::Vector{Float64}  where {T <: AbstractMatrix, G <: AbstractMatrix, Z <: AbstractArray}

    # @printf("Number of true objects: %d\n", length(truth_images))
    # @printf("Number of predicted objects: %d\n", length(prediction_images))

    thresholded_predictions = convert_to_binary(prediction_images,
                                                truth_images,
                                                τ)

    out = similar(τ)
    for k in eachindex(τ)
        l, p = thresholded_predictions[τ[k]]
        tp = sum( p .& l)
        fn = sum(.!p .& l)
        fp = sum( p .& .!l)
        out[k] = tp / (tp + fn + fp)
        # @printf("%2.2f   %d     %d      %d      %2.2f\n", τ[k], tp, fp, fn, out[k])
    end

    return out
end

function metric(prediction_images::Vector{T},
                truth_images::Vector{G};
                assert_intersection::Bool = false)::Float64 where {T <: AbstractMatrix, G <: AbstractMatrix}

    if assert_intersection
        assert_empty_intersection(prediction_images)
        assert_empty_intersection(truth_images)
    end

    τ_span = 0.5:0.05:0.95
    return mean(metric(prediction_images, truth_images, τ_span))
end

"Returns map: label -> vector of (linear index, length)"
function encode(this::Matrix{Int64})::Dict{Int64, Vector{Pair{Int64, Int64}}}
    labels = unique(this)
    labels = labels[labels .!= 0]

    out = Dict{Int64, Vector{Pair{Int64, Int64}}}()
    [out[x] = Vector{Pair{Int64, Int64}}(0) for x in labels]

    v = this[:]
    vals, lens = StatsBase.rle(v)
    idx = 0
    for k in eachindex(vals)
        label = v[idx + 1]
        @assert label == vals[k]
        enc = Pair(idx+1, lens[k])
        idx += lens[k]
        if !haskey(out, label)
            continue
        end

        push!(out[label], enc)
    end
    return out
end

function decode(this::Dict{Int64, Vector{Pair{Int64, Int64}}}, sz::Tuple{Int64, Int64})
    out = zeros(Int64, sz)
    out_vec = out[:]
    for (label, v) in this
        for item in v
            span = item.first : (item.first + item.second - 1)
            out_vec[span] .= label
        end
    end

    return reshape(out_vec, sz)
end

end
