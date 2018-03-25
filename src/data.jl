__precompile__(false)

module Data

using FileIO
using Images
using Juno
using ColorTypes
using FixedPointNumbers
using DataFrames
using CSVFiles
using DSB2018.Errors

using ImageView

global const STAGE1_TEST  = joinpath(@__DIR__, "../etc/stage1_test")
global const STAGE1_TRAIN = joinpath(@__DIR__, "../etc/stage1_train")

export Image, getTrainImageIds
export loadTrainImage
export getTrainImageMasks
export Prediction
export submission, submit!

function loadImage(id::String, stage_path::String)
    image_file = joinpath(stage_path, id, "images", id * ".png")
    return Images.load(image_file)
end

function loadMasks(id::String, stage_path::String)::Dict{String, Matrix{Bool}}
    out = Dict{String, Matrix{Bool}}()
    masks_root_path = joinpath(stage_path, id, "masks")
    for id in readdir(masks_root_path)
        mask_file = joinpath(masks_root_path, id)
        new_mask = Images.load(mask_file)
        out[id] = new_mask .> 0.0
        if isempty(find(out[id]))
            @show mask_file
            throw("Mask is empty")
        end
    end

    return out
end

struct Image{X}
    id::String
    image::Matrix{X}
    mask::Matrix{Int64}
    mask_labels::Vector{Int64}
end

struct Prediction
    id::String
    score::Float64
    mask::Matrix{Int64}
end

function loadTrainImage(id::String)::Image
    global STAGE1_TRAIN
    img = loadImage(id, STAGE1_TRAIN)
    masks = loadMasks(id, STAGE1_TRAIN)
    labeled_mask, labels = label_masks(collect(values(masks)))
    return Image(id, img, labeled_mask, labels)
end

function loadTestImage(id::String)::Image
    global STAGE1_TEST
    img = loadImage(id, STAGE1_TEST)
    return Image(id, img, zeros(Int64, size(img)), Int64[])
end

function getTrainImageIds()
    global STAGE1_TRAIN
    readdir(STAGE1_TRAIN)
end

function getTestImageIds()
    global STAGE1_TEST
    readdir(STAGE1_TEST)
end

function loadTrainImages(n::Int64=9999999999)
    ids = getTrainImageIds()
    n = min(n, length(ids))
    ids = ids[1:n]
    out = Vector{Image}(length(ids))
    Juno.@progress "Loading images" for k in eachindex(ids)
        out[k] = loadTrainImage(ids[k])
    end
    return out
end

function loadTestImages(n::Int64=999999999)
    ids = getTestImageIds()
    n = min(n, length(ids))
    out = Vector{Image}(length(ids))
    Juno.@progress "Loading images" for k in eachindex(ids)
        out[k] = loadTestImage(ids[k])
    end
    return out
end

function getTrainImageMasks(this::Image{X})::Vector{Matrix{Bool}} where {X <: Any}
    out = Vector{Matrix{Bool}}(length(this.mask_labels))
    for k in eachindex(out)
        out[k] = this.mask .== this.mask_labels[k]
    end

    return out
end

function label_masks(this::Vector{Matrix{Bool}})
    szs = size.(this)
    @assert length(unique(szs)) == 1
    sz = first(szs)
    out = zeros(Int64, sz)

    this_float = [x.*1.0 for x in this]
    num_p      = [sum(x) for x in this]

    s = sum(this_float)
    @assert maximum(s) == 1
    @assert minimum(num_p) > 0

    labels = Int64.(collect(1 : length(this)))
    for j in eachindex(this)
        out[this[j]] .= labels[j]
    end

    return out, labels
end

function encoded_string(this::Vector{Pair{Int64, Int64}})
    return join([join([x.first, x.second], " ") for x in this], " ")
end

function submission(this::Vector{Prediction})::DataFrame
    image_ids = Vector{String}(0)
    encodedpixels = Vector{String}(0)
    for prediction in this
        rle_encoded = Errors.encode(prediction.mask)
        for pairs::Vector{Pair{Int64, Int64}} in values(rle_encoded)
            push!(image_ids, prediction.id)
            push!(encodedpixels, encoded_string(pairs))
        end
    end

    return DataFrame(ImageId = image_ids, EncodedPixels = encodedpixels)
end

function submit!(this::DataFrame)
    submit_path = joinpath(@__DIR__, "../submissions")
    submit_file = string(now(Dates.UTC), ".csv")
    save(joinpath(submit_path, submit_file), this, quotechar = nothing)
end


end
