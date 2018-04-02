module Hills

using Flux
using Juno
using Plots
import Base: push!, append!
import Flux: train!

struct Container
    n::Int64
    data::Vector{Tuple{Vector{Float64}, Vector{Float64}}}
    function Container(N::Int64)
        return new(N, Vector{Tuple{Vector{Float64}, Vector{Float64}}}(0))
    end
end

function append!(this::Container,
                orig::Matrix{Float64},
                mask::Matrix{Float64}, can_invert::Bool = true)

    if can_invert
        if mean(orig) > 0.5
            orig = 1.0 - orig
        end
    end

    for k = 1 : size(orig, 1)
        n = min(this.n, size(orig, 2))
        sx = zeros(this.n)
        sy = zeros(this.n)

        sx = orig[k, 1 : n]
        sy = mask[k, 1 : n]

        push!(this, sx, sy)
    end

    for k = 1 : size(orig, 2)
        n = min(this.n, size(orig, 1))
        sx = zeros(this.n)
        sy = zeros(this.n)

        sx = orig[1 : n, k]
        sy = mask[1 : n, k]

        push!(this, sx, sy)
    end

    this
end

function push!(this::Container, x::Vector{Float64}, y::Vector{Float64})
    xn = maximum(x) == 0.0 ? x : x / maximum(x)
    yn = maximum(y) == 0.0 ? y : y / maximum(y)

    push!(this.data, (xn, yn))
end

function loss(x::Vector{Float64}, y::Vector{Float64}, mdl::Flux.Chain)
    return Flux.mse(mdl(x), y)
end

function loss(this::Tuple{Vector{Float64}, Vector{Float64}}, mdl::Flux.Chain)
    return loss(this[1], this[2], mdl)
end

function train(this::Container; mdl = get_model(this.n))
    numepochs = 1000
    opt = Flux.Nesterov(params(mdl), 0.5)

    ntrain = Int64(round(length(this.data)*0.5))
    jtrain = unique(rand(1:length(this.data), ntrain))
    jtest = setdiff(1:length(this.data), jtrain)

    train_losses = Vector{Tuple{Float64, Float64}}(0)
    test_losses  = Vector{Tuple{Float64, Float64}}(0)

    Juno.progress(name="Optimizing") do prog
        testdata  = this.data[jtest]
        traindata = this.data[jtrain]

        for k = 1 : numepochs
            Flux.train!((x, y) -> loss(x, y, mdl), traindata, opt)
            this_train_losses = [x.tracker.data for x in loss.(traindata, mdl)]
            this_test_losses = [x.tracker.data for x in loss.(testdata, mdl)]

            msg = @sprintf("%2.4f | %2.4f", mean(this_test_losses), std(this_test_losses))
            Juno.progress(prog, k / numepochs)
            Juno.right_text(prog, msg)

            push!(train_losses, (mean(this_train_losses), std(this_train_losses)))
            push!(test_losses, (mean(this_test_losses), std(this_test_losses)))

            tr_m = [x[1] for x in train_losses]
            tr_s = [x[2] for x in train_losses]
            p1 = scatter([tr_m tr_s], label = ["mean", "std"], title="train")

            ts_m = [x[1] for x in test_losses]
            ts_s = [x[2] for x in test_losses]
            p2 = scatter([ts_m ts_s], label = ["mean", "std"], title="test")

            display(plot(p1, p2, layout = grid(2, 1)))
        end
    end

    return mdl, train_losses, test_losses
end

function get_model(n::Int64)
    return Chain(Dense(n, n, Flux.σ),
                Dense(n, n, Flux.σ),
                Dense(n, n, Flux.σ),
                Dense(n, n, Flux.σ))
end

function filter_image(img::Matrix{Float64}, mdl::Chain, N::Int64)
    outx = zeros(size(img))
    outy = zeros(size(img))

    for k = 1 : size(img, 1)
        n = min(size(img, 2), N)
        outx[k, 1 : n] = mdl(img[k, 1 : n]).data
    end

    for k = 1 : size(img, 2)
        n = min(size(img, 1), N)
        outy[1 : n, k] = mdl(img[1 : n, k]).data
    end

    (outx, outy)
end

end
