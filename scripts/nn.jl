using DSB2018
using Flux
using ImageView
using Images
using Plots

function form_regressor(this::DSB2018.Data.Image)
    x̂ = Float64.(Gray.(this.image))

    dens = DSB2018.Model.to_density(x̂, 4)
    dens = abs.(dens) / sum(abs.(dens))

    adaptive_threshold = DSB2018.Model.skimage_filters.threshold_local(dens, 11, offset=0.0)
    global_threshold   = DSB2018.Model.skimage_filters.threshold_otsu(dens)

    w_adaptve = 0.1
    threshold = adaptive_threshold * w_adaptve .+ global_threshold * (1.0 - w_adaptve)

    img_t = dens .> threshold
    img_t = DSB2018.Model.skimage_morphology.remove_small_objects(img_t, min_size=4)
    img_t = DSB2018.Model.ndi.binary_fill_holes(img_t)
    img_f = Float64.(img_t)

    grad_y, grad_x, edges, orient = imedge(img_f, KernelFactors.ando3, "replicate")

    dens, img_f, edges
end

function form_regressand(this::DSB2018.Data.Image)
    ŷ = Float64.(this.mask .> 0)

    grad_y, grad_x, edges_hat, orient = imedge(ŷ, KernelFactors.ando3, "replicate")
    ŵ = copy(edges_hat).^0.5 + 1

    ŷ, ŵ
end

function form_prediction(model::Flux.Chain,
                            batch_size::Tuple{Int64, Int64},
                            img::DSB2018.Data.Image;
                            numsamples::Int64 = 100)

    dens, img_f, edges = form_regressor(img)
    batch_indices = DSB2018.Model.make_batches(size(dens), batch_size, 1)

    batch_indices = rand(batch_indices, numsamples)

    dens_batches  = DSB2018.Model.make_batches(dens, batch_indices)
    img_batches   = DSB2018.Model.make_batches(img_f, batch_indices)
    edges_batches = DSB2018.Model.make_batches(edges, batch_indices)

    res = zeros(size(img.image))
    num_visits = zeros(size(img.image))

    x = Array{Float64}((batch_size[1], batch_size[2], 3, 1))
    for k in eachindex(batch_indices)
        x[:, :, 1, 1] .= dens_batches[k]
        x[:, :, 2, 1] .= img_batches[k]
        x[:, :, 3, 1] .= edges_batches[k]
        prediction = model(x)

        idx = batch_indices[k]
        res[idx[1], idx[2]] .+= prediction.data
        num_visits[idx[1], idx[2]] .+= 1.0
    end

    res = res ./ num_visits
    res[num_visits .== 0.0] = 0.0

    N = sum(abs.(res))
    if N == 0.0
        return res
    end

    return res / N
end

function form_data(this::DSB2018.Data.Image,
                    batch_size::Tuple{Int64, Int64},
                    num::Int64)

    dens, img_f, edges = form_regressor(this)
    y, w = form_regressand(this)

    batch_indices = DSB2018.Model.make_batches(size(this.image), batch_size, 10)

    rnd_batch_indices = unique(rand(batch_indices, num))

    dens_batches   = DSB2018.Model.make_batches(dens, rnd_batch_indices)
    img_f_batches  = DSB2018.Model.make_batches(img_f, rnd_batch_indices)
    edges_batches  = DSB2018.Model.make_batches(edges, rnd_batch_indices)
    reg_batches    = DSB2018.Model.make_batches(y, rnd_batch_indices)
    weight_batches = DSB2018.Model.make_batches(w, rnd_batch_indices)

    num_batches = length(rnd_batch_indices)

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

    train_data
end

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

num_images_to_process = 30
batch_size = (128, 128)
num_samples_per_image = 10

ids = DSB2018.Data.getTrainImageIds()
XT = Tuple{Array{Float64,4},Array{Float64,2},Array{Float64,2}}
train_data = Vector{XT}(0)
for k = 1 : num_images_to_process
    img = DSB2018.Data.loadTrainImage(ids[k])
    append!(train_data, form_data(img, batch_size, num_samples_per_image))
end

all_images = DSB2018.Data.loadTrainImage.(ids)

function loss(this::Tuple) #train_data
    return loss(this[1], this[2], this[3])
end

function loss(x, y, w)
    r = nn_model(x)
    δ = (r .- y) .* w
    z = sum(δ.^2)/length(y)
    (σ.(z/10) - 0.5) / 0.005*0.02
end

bx = batch_size[1]
by = batch_size[2]

struct IMResize2
    size::Tuple{Int64, Int64}
end

Flux.treelike(IMResize2)
function (this::IMResize2)(x)
    return Flux.Tracker.track(Images.imresize, x, this.size)
end

import Flux.Tracker.back

function back(::typeof(Images.imresize), Δ, xs::Flux.TrackedArray, _...)
    back(xs, Images.imresize(Δ, size(xs)))
end

rz_down1 = IMResize2((88, 88))
rz_down2 = IMResize2((56, 56))

rz_up1   = IMResize2((88, 88))
rz_up2   = IMResize2((bx, by))

nn_model = Chain(Flux.Conv((9, 9), 3 => 1, pad = (4, 4), σ),
                x -> reshape(x, (bx, by)),
                x -> rz_down1(x),
                x -> reshape(x, (88, 88, 1, 1)),
                Flux.Conv((9, 9), 1 => 1, pad = (4, 4), σ),
                x -> reshape(x, (88, 88)),
                x -> rz_down2(x),
                x -> reshape(x, (56, 56, 1, 1)),
                Flux.Conv((9, 9), 1 => 1, pad = (4, 4), σ),
                x -> reshape(x, (56, 56)),
                x -> rz_up1(x),
                x -> reshape(x, (88, 88, 1, 1)),
                Flux.Conv((9, 9), 1 => 1, pad = (4, 4), σ),
                x -> reshape(x, (88, 88)),
                x -> rz_up2(x)
                )

opt = Flux.Nesterov(params(nn_model), 0.2)
train_percent = 0.6

ltest  = Float64[]
ltrain = Float64[]

num_train = Int64(round(length(train_data)*train_percent))
num_test  = length(train_data) - num_train

jtrain = unique(rand(1 : length(train_data), num_train))
jtest  = setdiff(1:length(train_data), jtrain)

check_points = Vector{Flux.Chain}(0)

Juno.@progress "Training" for k = 1 : 3000
    Flux.train!(loss, train_data[jtrain], opt)
    @show k

    train_loss = mean(loss.(train_data[jtrain]))
    test_loss = mean(loss.(train_data[jtest]))

    push!(ltrain, train_loss.tracker.data)
    push!(ltest, test_loss.tracker.data)
    push!(check_points, deepcopy(nn_model))

    if (mod(k, 10) == 0) || (k == 1)
        plt_train = plot(ltrain[1:end],
                        title = @sprintf("Train Loss: %2.6f", train_loss),
                        fillrange = [minimum(ltrain) maximum(ltrain)],
                        fillalpha = 0.3)

        plt_valid = plot(ltest[1:end],
                        title = @sprintf("Test Loss: %2.6f", test_loss),
                        fillrange = [minimum(ltest) maximum(ltest)],
                        fillalpha = 0.3)

        display(plot(plt_train, plt_valid, layout = grid(2, 1)))
    end
end

scores = Vector{Float64}(length(all_images))
Juno.@progress "Metric" for k in eachindex(all_images)
    scores[k] = get_metric(all_images[k],
                        form_prediction(nn_model, batch_size, all_images[k], numsamples = 90))
    display(scatter(scores[1:k], title=@sprintf("%2.4f", mean(scores[1:k]))))
end

scores_prev = Vector{Float64}(length(all_images))
Juno.@progress "Metric2" for k in eachindex(scores_prev)
    x̂ = Float64.(Gray.(all_images[k].image))
    p̂ = DSB2018.Model.to_density(x̂, 4)
    scores_prev[k] = get_metric(all_images[k], p̂)
end


scatter(scores, title=@sprintf("%2.4f", mean(scores)))
scatter(scores_prev, title=@sprintf("%2.4f", mean(scores_prev)))

imshow(pred)
get_metric(img, pred)

x̂ = Float64.(Gray.(img.image))
