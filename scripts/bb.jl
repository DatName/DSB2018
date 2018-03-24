using BlackBoxOptim

function get_metric(this::DSB2018.Data.Image, res::Matrix{Float64})

    adaptive_threshold = DSB2018.Model.skimage_filters.threshold_local(res, 11, offset=0.0)
    global_threshold   = DSB2018.Model.skimage_filters.threshold_otsu(res)

    w_adaptve = 0.1
    threshold = adaptive_threshold * w_adaptve .+ global_threshold * (1.0 - w_adaptve)

    img_t = res .> threshold
    img_t = DSB2018.Model.skimage_morphology.remove_small_objects(img_t, min_size=10)
    img_t = DSB2018.Model.ndi.binary_fill_holes(img_t)

    labeled_prediction = Images.label_components(img_t)
    prediction_vector  = DSB2018.Model.split_labeled_matrix(labeled_prediction)
    truth_images = getTrainImageMasks(this)

    return metric(prediction_vector, truth_images)
end

ids = DSB2018.Data.getTrainImageIds()
img = DSB2018.Data.loadTrainImage(ids[1])

x̂ = Float64.(Gray.(img.image))
ŷ = Float64.(img.mask .> 0)

function mask(c::T, x::Float64, y::Float64) where {T}
    μx = c[1]
    μy = c[2]
    σx = c[3]
    σy = c[4]
    ρ  = c[5]
    w  = c[6]

    h = (x - μx)^2.0/σx^2. + (y - μy)^2.0/σy^2. - 2.*ρ*(x-μx)*(y-μy)/σx/σy
    w * exp( -h/2./(1.0 - ρ^2.) )
end

function bbpredict(c::Vector{Float64}, sz::Tuple{Int64, Int64})
    τ = Matrix{Float64}(sz)
    r = CartesianRange(CartesianIndex(1, 1), CartesianIndex(size(τ)))
    for iz in r
        x, y = iz[1]/sz[1], iz[2]/sz[2]
        τ[iz] = mask(c, x, y)
    end
    τ
end

function bbpredict!(c::Vector{Float64}, τ::Matrix{Float64})
    sz = size(τ)
    r = CartesianRange(CartesianIndex(1, 1), CartesianIndex(size(τ)))
    for iz in r
        x, y = iz[1]/sz[1], iz[2]/sz[2]
        τ[iz] = mask(c, x, y)
    end
    τ
end

c = [0.5, 0.1, 0.3, 0.6, 0.9, 1.]

function bbloss(c::Vector{Float64}, img::Matrix{Float64})
    out = 0.0
    for iz in CartesianRange(CartesianIndex(1, 1), CartesianIndex(size(img)))
        x, y = iz[1]/size(img)[1], iz[2]/size(img)[2]
        out += (img[iz] - mask(c, x, y)).^2
    end
    out / mean(img).^2
end

sr = Vector{Tuple{Float64, Float64}}(6)
sr[1] = (0, 1)
sr[2] = (0, 1)
sr[3] = (0.001, 0.05)
sr[4] = (0.001, 0.05)
sr[5] = (-0.5, 0.5)
sr[6] = (0.001, 2.0)

res = bboptimize(x -> bbloss([c0[1:2]; x], x̂),
                NumDimensions = 4,
                SearchRange = sr,
                MaxSteps = 100000,
                PopulationSize = 10,
                Method = :random_search)

c0 = best_candidate(res)
imshow(bbpredict(c0, size(ŷ)))
imshow(x̂)
