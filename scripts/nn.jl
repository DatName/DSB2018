using DSB2018
using Flux
using ImageView
using Images
using Plots


function form_regressor(this::DSB2018.Data.Image)
    x̂ = Float64.(Gray.(this.image))

    ∂xx, ∂xy = imgradients(x̂, KernelFactors.ando3)
    dens = DSB2018.Model.to_density(x̂, 4)

    grad_y, grad_x, edges, orient = imedge(dens, KernelFactors.ando3, "replicate")

    adaptive_threshold = DSB2018.Model.skimage_filters.threshold_local(dens, 11, offset=0.0)
    global_threshold   = DSB2018.Model.skimage_filters.threshold_otsu(dens)

    w_adaptve = 0.1
    threshold = adaptive_threshold * w_adaptve .+ global_threshold * (1.0 - w_adaptve)

    img_t = dens .> threshold
    img_t = DSB2018.Model.skimage_morphology.remove_small_objects(img_t, min_size=4)
    img_t = DSB2018.Model.ndi.binary_fill_holes(img_t)
    img_f = Float64.(img_t)

    dens, img_f, edges
end

function form_regressand(this::DSB2018.Data.Image)
    ŷ = Float64.(this.mask .> 0)

    grad_y, grad_x, edges_hat, orient = imedge(ŷ, KernelFactors.ando3, "replicate")
    ŵ = copy(edges_hat).^0.5 + 1

    ŷ, ŵ
end

function form_data(this::DSB2018.Data.Image,
                    batch_size::Tuple{Int64, Int64},
                    num::Int64)

    dens, img_f, edges = form_regressor(this)
    y, w = form_regressand(this)

    dens_batches_all  = DSB2018.Model.make_batches(dens, batch_size, δ = 10)

    j = unique(rand(1 : length(dens_batches_all), num))

    dens_batches = dens_batches_all[j]

    img_f_batches = DSB2018.Model.make_batches(img_f, batch_size, δ = 10)[j]
    edges_batches = DSB2018.Model.make_batches(edges, batch_size, δ = 10)[j]
    reg_batches   = DSB2018.Model.make_batches(y, batch_size, δ = 10)[j]
    weight_batches = DSB2018.Model.make_batches(w, batch_size, δ = 10)[j]

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

num_images_to_process = 10
batch_size = (128, 128)
num_samples_per_image = 10

ids = DSB2018.Data.getTrainImageIds()
XT = Tuple{Array{Float64,4},Array{Float64,2},Array{Float64,2}}
train_data = Vector{XT}(0)
for k = 1 : num_images_to_process
    img = DSB2018.Data.loadTrainImage(ids[k])
    append!(train_data, form_data(img, batch_size, num_samples_per_image))
end

function loss(this::Tuple) #train_data
    return loss(this[1], this[2], this[3])
end

function loss(x, y, w)
    r = nn_model(x)
    δ = (r .- y) .* w
    z = sum(δ.^2)/length(y)
    (σ.(z/10) - 0.5) / 0.005*0.02
end

nn_model = Chain(Flux.Conv((9, 9), 3 => 1, pad = (4, 4), σ),
                x -> reshape(x, (batch_size[1], batch_size[2])))

opt = Flux.Nesterov(params(nn_model), 0.2)
train_percent = 0.5

ltest  = Float64[]
ltrain = Float64[]

num_train = Int64(round(length(train_data)*train_percent))
num_test  = length(train_data) - num_train

jtrain = rand(1 : length(train_data), num_train)
jtest  = rand(1 : length(train_data), num_test)

Juno.@progress "Training" for k = 1 : 5000
    Flux.train!(loss, train_data[jtrain], opt)

    train_loss = mean(loss.(train_data[jtrain]))
    test_loss = mean(loss.(train_data[jtest]))

    push!(ltrain, train_loss.tracker.data)
    push!(ltest, test_loss.tracker.data)

    if mod(k, 10) == 0
        plt_train = plot(ltrain[10:end],
                        title = @sprintf("Train Loss: %2.6f", train_loss),
                        fillrange = [minimum(ltrain) maximum(ltrain)],
                        fillalpha = 0.3)

        plt_test = plot(ltest[10:end],
                        title = @sprintf("Test Loss: %2.6f", test_loss),
                        fillrange = [minimum(ltest) maximum(ltest)],
                        fillalpha = 0.3)

        display(plot(plt_train, plt_test, layout = grid(2, 1)))
    end
end
