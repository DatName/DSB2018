module Data

using FileIO
using Images
using Juno
using DSB2018.Errors

import DSB2018.Errors: metric
using ImageView
import ImageView: imshow

global const STAGE1_TEST  = joinpath(@__DIR__, "../etc/stage1_test")
global const STAGE1_TRAIN = joinpath(@__DIR__, "../etc/stage1_train")

const IMG = Matrix{ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}}}
const MIMG = BitArray{2}

export getTrainImage, loadTrainImages

struct Image
    id::String
    image::IMG
    masks::Dict{String, MIMG}
    function Image(id::String, root_path::String)
        images = joinpath(root_path, id, "images")
        masks  = joinpath(root_path, id, "masks")
        images_files = readdir(images)
        if length(images_files) != 1
            throw(ErrorException("Not unique image"))
        end

        image = Images.load(joinpath(images, first(images_files)))
        masks_objects = Dict{String, IMG}()
        for item in readdir(masks)
            new_mask = Images.load(joinpath(masks, item))
            @assert length(unique(new_mask)) == 2
            @assert isempty(setdiff(unique(new_mask), [0, 1]))

            m = BitArray(size(new_mask))
            m[new_mask .== 0] = false
            m[new_mask .== 1] = true
            masks_objects[item] = m
        end

        return new(id, image, masks_objects)
    end
end

global const TrainImages = Dict{String, Image}()
global const TestImages = Dict{String, Image}()

function loadTrainImages(atmost::Float64 = Inf)
    global TrainImages
    global STAGE1_TRAIN
    c = 0.0
    Juno.@progress "Loading train images" for image_id in readdir(STAGE1_TRAIN)
        TrainImages[image_id] = Image(image_id, STAGE1_TRAIN)
        c += 1.0
        if c > atmost
            break
        end
    end
end

function getTrainImage(this::Image)
    m = zeros(RGB{Float64}, size(this.image))

    for mask in values(this.masks)
        c = RGB(rand(), rand(), rand())
        for k in eachindex(mask)
            if !mask[k]
                continue
            end
            
            r, n = ind2sub(size(m), k)
            idx = CartesianIndex{2}(r, n)
            m[idx] = c
        end
    end

    return m
end


end
