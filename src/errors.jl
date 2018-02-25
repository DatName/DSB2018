module Errors

using MLBase

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
                τ::Z)::Tuple{Matrix{Bool}, Vector{Bool}} where {T <: AbstractMatrix, G <: AbstractMatrix, Z <: AbstractArray}

    prediction = Vector{Bool}(length(prediction_images))
    labels     = Matrix{Bool}(length(prediction), length(τ))

    truth_image_areas = Dict{Int64, Area}()
    [truth_image_areas[x] = non_zero_area(truth_images[x]) for x in eachindex(truth_images)]

    for k in eachindex(prediction_images)
        img = prediction_images[k]
        img_area = non_zero_area(img)
        score = -Inf

        for (idx, truth_area) in truth_image_areas
            score = intersection_over_union(img_area, truth_area)
            if score > 0.0
                delete!(truth_image_areas, idx)
                break
            end
        end

        prediction[k] = true
        labels[k, :] = score .> τ
    end

    for idx in truth_image_areas
        push!(prediction, false)
        labels = [labels; trues(1, length(τ))]
    end

    return (labels, prediction)
end

"Result for a single image"
function metric(prediction_images::Vector{T},
                truth_images::Vector{G},
                τ::Z)::Vector{Float64}  where {T <: AbstractMatrix, G <: AbstractMatrix, Z <: AbstractArray}

    labels, predictions = convert_to_binary(prediction_images,
                                            truth_images,
                                            τ)

    out = similar(τ)
    for k in eachindex(τ)
        r = MLBase.roc(labels[:, k], predictions)
        out[k] = r.tp / (r.tp + r.fp + r.fn)
    end

    return out
end

function metric(prediction_images::Vector{T},
                truth_images::Vector{G}; assert_intersection::Bool = false)::Float64 where {T <: AbstractMatrix, G <: AbstractMatrix}

    if assert_intersection
        assert_empty_intersection(prediction_images)
        assert_empty_intersection(truth_images)
    end

    τ_span = 0.5:0.05:0.95
    return mean(metric(prediction_images, truth_images, τ_span))
end

end
