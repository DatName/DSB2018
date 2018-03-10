module Data

using FileIO
using Images
using Juno
using ColorTypes
using FixedPointNumbers

using ImageView

global const STAGE1_TEST  = joinpath(@__DIR__, "../etc/stage1_test")
global const STAGE1_TRAIN = joinpath(@__DIR__, "../etc/stage1_train")

export getTrainImage, loadTrainImages!
export getTrainImageIds
export loadTrainImage

struct Image{X}
    id::String
    image::Matrix{X}
    masks::Dict{String, Matrix{Bool}}
    function Image{X}(id::String, root_path::String; assert_masks::Bool = false) where {X <: Any}
        images = joinpath(root_path, id, "images")
        masks  = joinpath(root_path, id, "masks")
        images_files = readdir(images)
        if length(images_files) != 1
            throw(ErrorException("Not unique image"))
        end

        image = Images.load(joinpath(images, first(images_files)))
        masks_objects = Dict{String, Matrix{Bool}}()
        u = oneunit(ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}})
        for item in readdir(masks)
            new_mask = Images.load(joinpath(masks, item))
            if assert_masks
                @assert length(unique(new_mask)) == 2
                @assert isempty(setdiff(unique(new_mask), [0, 1]))
            end
            masks_objects[item] = new_mask .== u
        end

        return new{X}(id, image, masks_objects)
    end
end

global const TrainImages = Dict{String, Image}()
global const TestImages = Dict{String, Image}()

function getTrainImageIds()
    global STAGE1_TRAIN
    readdir(STAGE1_TRAIN)
end

function loadTrainImage(id::String)::Image{Gray{Normed{UInt8,8}}}
    global STAGE1_TRAIN
    return Image{Gray{Normed{UInt8,8}}}(id, STAGE1_TRAIN)
end

function loadTrainImages!(atmost = Inf)
    global TrainImages
    global STAGE1_TRAIN
    c = 0.0
    Juno.@progress "Loading train images" for image_id in readdir(STAGE1_TRAIN)
        TrainImages[image_id] = Image{Gray{Normed{UInt8,8}}}(image_id, STAGE1_TRAIN)
        c += 1.0
        c > atmost &&  break
    end

    return TrainImages
end

function truth_image(this::Image{X})::Matrix{Bool} where {X <: Any}
    out = falses(size(this.image))
    for mask in values(this.masks)
        for x in CartesianRange(CartesianIndex(1, 1), CartesianIndex(size(out)))
            if mask[x]
                out[x] = true
            end
        end
    end

    return out
end

end
