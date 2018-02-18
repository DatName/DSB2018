module Data

using FileIO
using Images
using Juno
using DSB2018.Errors
using ColorTypes
using FixedPointNumbers

import DSB2018.Errors: metric
using ImageView
import ImageView: imshow

global const STAGE1_TEST  = joinpath(@__DIR__, "../etc/stage1_test")
global const STAGE1_TRAIN = joinpath(@__DIR__, "../etc/stage1_train")

export getTrainImage, loadTrainImages!
export showtruth, test_image, true_percent

struct Image{X}
    id::String
    image::Matrix{X}
    masks::Dict{String, BitArray{2}}
    function Image{X}(id::String, root_path::String) where {X <: Any}
        images = joinpath(root_path, id, "images")
        masks  = joinpath(root_path, id, "masks")
        images_files = readdir(images)
        if length(images_files) != 1
            throw(ErrorException("Not unique image"))
        end

        image = Images.load(joinpath(images, first(images_files)))
        masks_objects = Dict{String, BitArray{2}}()
        for item in readdir(masks)
            new_mask = Images.load(joinpath(masks, item))
            @assert length(unique(new_mask)) == 2
            @assert isempty(setdiff(unique(new_mask), [0, 1]))

            m = BitArray(size(new_mask)) .* false
            for x in CartesianRange(CartesianIndex(1, 1), CartesianIndex(size(m)))
                m[x] = new_mask[x] == 1
            end

            masks_objects[item] = m
        end

        return new{X}(id, image, masks_objects)
    end
end

global const TrainImages = Dict{String, Image}()
global const TestImages = Dict{String, Image}()

function loadTrainImages!(atmost::Float64 = Inf)
    global TrainImages
    global STAGE1_TRAIN
    c = 0.0
    Juno.@progress "Loading train images" for image_id in readdir(STAGE1_TRAIN)
        TrainImages[image_id] = Image{Gray{Normed{UInt8,8}}}(image_id, STAGE1_TRAIN)
        c += 1.0
        c > atmost &&  break
    end
end

function test_image(this::Image{X})::BitArray{2} where {X <: Any}
    out = BitArray(size(this.image)) .* false

    for mask in values(this.masks)
        for x in CartesianRange(CartesianIndex(1, 1), CartesianIndex(size(out)))
            if mask[x]
                out[x] = true
            end
        end
    end

    return out
end

function true_percent(this::BitArray{2})::Float64
    return sum(this) / prod(size(this))
end

function showtruth(this::Image{X}) where {X <: Any}
    out = Matrix{RGB{Float64}}(size(this.image))
    for x in CartesianRange(CartesianIndex(1, 1), CartesianIndex(size(out)))
        out[x] = RGB(0.0, 0.0, 0.0)
    end

    for mask in values(this.masks)
        clr = RGB(rand(), rand(), rand())
        for x in CartesianRange(CartesianIndex(1, 1), CartesianIndex(size(out)))
            if mask[x]
                out[x] = clr
            end
        end
    end

    imshow(out)
end

end
