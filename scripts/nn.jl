using DSB2018
using Flux
using ImageView
using Images
using Plots

ids = DSB2018.Data.getTrainImageIds()
img = DSB2018.Data.loadTrainImage(ids[1])

x̂ = Float64.(Gray.(img.image))
ŷ = Float64.(img.mask .> 0)

grad_y, grad_x, ŷ, orient = imedge(ŷ, KernelFactors.ando3, "replicate")

∂xx, ∂xy = imgradients(x̂, KernelFactors.ando3)
dens = DSB2018.Model.to_density(x̂, 4)

adaptive_threshold = DSB2018.Model.skimage_filters.threshold_local(dens, 41, offset=0.0)
global_threshold   = DSB2018.Model.skimage_filters.threshold_otsu(dens)

w_adaptve = 0.5
threshold = adaptive_threshold * w_adaptve .+ global_threshold * (1.0 - w_adaptve)

img_t = dens .> threshold
img_t = DSB2018.Model.skimage_morphology.remove_small_objects(img_t, min_size=4)
img_t = DSB2018.Model.ndi.binary_fill_holes(img_t)
img_f = Float64.(img_t)

nx, ny = size(x̂)

target = Array{Float64}((nx, ny, 4, 1))
target[:, :, 1, 1] = dens
target[:, :, 2, 1] = ∂xx
target[:, :, 3, 1] = ∂xy
target[:, :, 4, 1] = img_f

w = Flux.param(rand())
μ = Flux.param(rand())

ft = typeof(w)
function τ_gate(x::T, xw::ft, xμ::ft) where {T}
    tanh.((x .- xμ).*xw)/2.0 .+ 0.5
end

nn_model = Chain(Flux.Conv((5, 5), 4 => 2, pad = (2, 2)),
                Flux.Conv((3, 3), 2 => 1, pad = (1, 1)),
                x -> reshape(x, (size(x)[1], size(x)[2]))
                )

imshow(nn_model(target).data)
# imshow(ŷ)

function loss(x, y)
    Flux.mse(nn_model(x), y)
end

opt = Flux.Nesterov(params(nn_model), 0.1)
loss(target, ŷ)

l = Float64[]
Juno.@progress "Training" for k = 1 : 1000
    Flux.train!(loss, [(target, ŷ)], opt)
    newl = loss(target, ŷ)
    push!(l, newl.tracker.data)
    if mod(k, 10) == 0
        p2 = plot(l, title=string(newl), fillrange = [(minimum(l) - 0.000)*ones(size(l)) l])
        display(p2)
    end
end

@benchmark nn_model(target) #95 ms
imshow(nn_model(target).data)
