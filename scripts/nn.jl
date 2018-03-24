using DSB2018
using Flux
using ImageView
using Images
using Plots


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

grad_y, grad_x, edges_hat, orient = imedge(ŷ, KernelFactors.ando3, "replicate")
ŵ = copy(edges_hat).^0.5 + 1

∂xx, ∂xy = imgradients(x̂, KernelFactors.ando3)
dens = DSB2018.Model.to_density(x̂, 4)

grad_y, grad_x, edges, orient = imedge(dens, KernelFactors.ando3, "replicate")

adaptive_threshold = DSB2018.Model.skimage_filters.threshold_local(dens, 41, offset=0.0)
global_threshold   = DSB2018.Model.skimage_filters.threshold_otsu(dens)

w_adaptve = 0.5
threshold = adaptive_threshold * w_adaptve .+ global_threshold * (1.0 - w_adaptve)

img_t = dens .> threshold
img_t = DSB2018.Model.skimage_morphology.remove_small_objects(img_t, min_size=4)
img_t = DSB2018.Model.ndi.binary_fill_holes(img_t)
img_f = Float64.(img_t)

##########################################################################################
T̂ = Array{Float64, 4}(size(dens, 1), size(dens, 2), 3, 1)
T̂[:, :, 1, 1] = dens
T̂[:, :, 2, 1] = img_f
T̂[:, :, 3, 1] = edges

batch_size = (128, 128)
δ = 30

dens_batches  = DSB2018.Model.make_batches(dens, batch_size, δ = δ)
img_f_batches = DSB2018.Model.make_batches(img_f, batch_size, δ = δ)
edges_batches = DSB2018.Model.make_batches(edges, batch_size, δ = δ)
reg_batches   = DSB2018.Model.make_batches(ŷ, batch_size, δ = δ)
weight_batches = DSB2018.Model.make_batches(ŵ, batch_size, δ = δ)

num_batches = length(reg_batches)

target = Array{Float64}((batch_size[1], batch_size[2], 3, num_batches))
for k = 1 : num_batches
    target[:, :, 1, k] = dens_batches[k]
    target[:, :, 2, k] = img_f_batches[k]
    target[:, :, 3, k] = edges_batches[k]
end

train_data = Vector{Tuple{Array{Float64, 4}, Matrix{Float64}, Matrix{Float64}}}(0)
for k = 1 : num_batches
    t = Array{Float64, 4}((size(target)[1], size(target)[2], size(target)[3], 1))
    t[:, :, :, 1] = target[:, :, :, k]

    push!(train_data, (t, reg_batches[k], weight_batches[k]))
end

function loss(this::Tuple)
    return loss(this[1], this[2], this[3])
end

function loss(x, y, w)
    r = nn_model(x)
    δ = (r .- y) .* w
    z = sum(δ.^2)/length(y)
    (σ.(z/10) - 0.5) / 0.005*0.02
end

nn_model = Chain(Flux.Conv((5, 5), 3 => 1, pad = (2, 2), σ),
                x -> reshape(x, (batch_size[1], batch_size[2])))

nn_model(train_data[1][1])
loss(train_data[1])

opt = Flux.Nesterov(params(nn_model), 0.3)

base_metric = get_metric(img, dens)

l = Float64[]
p = Float64[]
newp = NaN

Juno.@progress "Training" for k = 1 : 5000
    Flux.train!(loss, train_data[1 : 20], opt)
    check_data = train_data[ 20 : end]
    newl = 0.0
    for d in check_data
        newl += loss(d)
    end
    newl = newl / length(check_data)

    push!(l, newl.tracker.data)

    if mod(k, 10) == 0
        plt = plot(l[30:end], title=@sprintf("Loss: %2.6f", newl), fillrange = [minimum(l) maximum(l)], fillalpha = 0.3)
        display(plt)
    end
end

DSB2018.Model.apply_batched_model(nn_model, batch_size, T̂)

imshow(nn_model(target).data)

r = nn_model(target)
