module Errors

using MLBase

const Area = Vector{CartesianIndex{2}}

export metric

function intersection_over_union(this::Matrix{Bool}, that::Matrix{Bool})::Float64
    ni = sum(this .* that)
    nu = sum(this .| that)
    return ni / nu
end

function intersection_over_union(this::Area, that::Area)::Float64
    ni = length(this ∩ that)
    nu = length(this ∪ that)
    return ni / nu
end

function non_zero_area(this::T)::Vector{CartesianIndex{2}} where {T <: AbstractMatrix}
    linear_indices = find(this)
    out = Vector{CartesianIndex{2}}(length(linear_indices))
    for k in eachindex(linear_indices)
        r, c = ind2sub(size(this), linear_indices[k])
        out[k] = CartesianIndex{2}(r, c)
    end
    return out
end

function assert_empty_intersection(this::Vector{T})::Void where {T <: AbstractMatrix}
    covered_area = Vector{CartesianIndex{2}}(0)
    for img in this
        area = non_zero_area(img)
        @assert isempty(intersect(area, covered_area))
        append!(covered_area, area)
    end

    return nothing
end

function convert_to_binary(prediction_images::Vector{T},
                truth_images::Vector{G},
                τ::Float64)::Tuple{Vector{Bool}, Vector{Bool}} where {T <: AbstractMatrix, G <: AbstractMatrix}

    prediction = Vector{Bool}(length(prediction_images))
    label = similar(prediction)

    truth_image_areas = Dict{Int64, Vector{CartesianIndex{2}}}()
    [truth_image_areas[x] = non_zero_area(truth_images[x]) for x in eachindex(truth_images)]

    for k in eachindex(prediction_images)
        img = prediction_images[k]
        img_area = non_zero_area(img)
        score = -Inf

        for (idx, truth_area) in truth_image_areas
            if !isempty(intersect(img_area, truth_area))
                score = intersection_over_union(img_area, truth_area)
                delete!(truth_image_areas, idx)
                break
            end
        end

        prediction[k] = true
        label[k] = score > τ
    end

    for idx in truth_image_areas
        push!(prediction, false)
        push!(label, true)
    end

    return (label, prediction)
end

"Result for a single image"
function metric(prediction_images::Vector{T},
                truth_images::Vector{G},
                τ::Float64)::Float64  where {T <: AbstractMatrix, G <: AbstractMatrix}

    labels, predictions = convert_to_binary(prediction_images,
                                            truth_images,
                                            τ)

    r = MLBase.roc(labels, predictions)

    return r.tp / (r.tp + r.fp + r.fn)
end

function metric(prediction_images::Vector{T},
                truth_images::Vector{G})::Float64 where {T <: AbstractMatrix, G <: AbstractMatrix}

    assert_empty_intersection(prediction_images)
    assert_empty_intersection(truth_images)

    τ_span = 0.5:0.05:0.95
    m = 0.0
    for τ in τ_span
        m += metric(prediction_images, truth_images, τ)
    end

    return m / length(τ_span)
end

end
