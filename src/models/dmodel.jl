using OffsetArrays
using ImageFiltering
using ImageFiltering.Kernel
using ImageSegmentation

mutable struct XModel
    size::Int64
    σ::Float64

    base_color::Float64
    kernel::OffsetArray
    range::StepRange{Int64, Int64}
    function XModel(sz::Int64,
                    σ::Float64)
        return new(sz, σ,
                    NaN, isnan(σ) ?  OffsetArray{Float64}(0:0) : ImageFiltering.Kernel.gaussian(σ), -sz:1:sz)
    end
end

function fit!(this::XModel, m::Matrix{Float64})
    nr, nc = size(m)
    for k = 1 : nr        pline = m[k, :]
    end
end

function borders(img::Matrix{Float64}, τ::Float64)
    gx, gy = imgradients(img)
    mag, grad_angle = magnitude_phase(gx, gy)
    mag[mag .< τ] = 0.0  # Threshold magnitude image
    thinned, subpix =  thin_edges_subpix(mag, grad_angle)
    return thinned
end

function borders_2(img::Matrix{Float64}, fmin::Float64, fmax::Float64, σ::Float64, n::Int64, h::Int64)
    fscale = scaleminmax(Float64, fmin, fmax);
    canny_edges = canny(fscale.(img), (Percentile(n), Percentile(h)), σ)
    return canny_edges
end

function magic_transform(this::Matrix{X}) where {X <: Any}
    return abs(2.0 * (this .- 0.5))
end

function start!(this::Colony{XModel, X}, poligon::Area{X}, sz::Tuple{Int64, Int64})::Coordinate where {X <: Any}

    img = convert(Matrix{Float64}, poligon, sz)
    # img = clahe(img, 20, xblocks = 8, yblocks = 8, clip = 3)

    # imfilter(img, Kernel.DoG(1.))
    # a, b = imgradients(img, KernelFactors.sobel, "replicate")
    # m    = Images.magnitude(a, b)
    # morphogradient(img)
    #NB:2
    # hist_equalised_img = clahe(img, 20, xblocks = 8, yblocks = 8, clip = 3)
    # img  = Images.imROF(img, 0.2, 20)

    if !isnan(this.model.σ)
        img = imfilter(img, ImageFiltering.Kernel.gaussian(this.model.σ))
    end

    if isnan(this.model.base_color)
        v = collect(values(poligon))
        qmax = maximum(v)
        qmin = minimum(v)
        this.model.base_color = qmin + (qmax * 0.99 - qmin) / 6.
        # this.model.base_color = (maximum(v) - mean(v)) / maximum(v)
    end

    #replace, not add
    for (k, v) in poligon
        poligon[k] = X == Bool ? img[k] > this.model.base_color : img[k]
    end

    v = collect(values(poligon))
    x = collect(keys(poligon))

    v0, i0 = findmax(v)
#
    return x[i0]
end
#
function next!(this::Colony{XModel, X},
                poligon::Area{X},
                imagesize::Tuple{Int64, Int64})::Union{Void, Coordinate} where {X <: Any}

    if isempty(poligon)
        return nothing
    end

    c = convert(Matrix{X}, poligon, imagesize)
    vmax, imax = findmax(c)
    imax_c = CartesianIndex(ind2sub(imagesize, imax))
    r = collect(CartesianRange(CartesianIndex(1,1), CartesianIndex(imagesize)))
    d = distance.(imax_c, r);
    d = (d + minimum(d)) / maximum(d)

    n = d .* (c - vmax)

    v0, i0 = findmax(n)
    ci0 = CartesianIndex(ind2sub(imagesize, i0))

    # mv = mean(this)
    m0 = c[ci0]

    if m0 < this.model.base_color
        return nothing
    else
        return ci0
    end
end

function done!(this::Colony{XModel, X}, poligon::Area{X}, x::Union{Void, Coordinate})::Bool where {X <: Any}
    return x == nothing
end

function belongs(this::Cell{XModel, Bool},
                cur::Coordinate,
                nxt::Pair{Coordinate, Bool})::Bool

    if !nxt.second
        return false
    end

    return distance(cur, nxt.first) < this.model.size
end

function belongs(this::Cell{XModel, X},
                cur::Coordinate,
                nxt::Pair{Coordinate, X})::Bool where {X <: Any}

    if distance(cur, nxt.first) > this.model.size
        return false
    end

    base_color = this.model.base_color
    nxt_color = nxt.second

    return nxt_color >= base_color
end

function iscell(this::Cell{XModel, X})::Bool where {X <: Any}
    return length(this.area) > this.model.size
end

"make cell appear as white"
function make_top_white(this::Matrix{X})::Matrix{X} where {X <: Any}
    mn, mx = extrema(this)
    md = mean(this[:])

    if sum(this .> md) < sum(this .< md)
        return copy(this)
    else
        return abs.(this - 1.0)
    end
end

function make_top_white(this::Matrix{Bool}; qlow::Float64 = 0.1, qhigh::Float64 = 0.9)::Matrix{Bool}
    x1 = sum(this)
    x2 = sum(.!this)

    if x2 > x1
        return .!this
    end

    return copy(this)
end

function grow(input::Matrix{X},
            input_model::XModel; show::Bool = false)::Colony{XModel, Bool} where {X <: Any}
    model = deepcopy(input_model)
    this = make_top_white(input)
    # imshow(this)

    grey_image   = this
    grey_poligon = convert(Area{Float64}, grey_image)
    grey_colony  = grow!(Colony(model, Float64), grey_poligon, size(grey_image), show = show)

    # imshow(grey_colony, size(grey_image))
    black_image  = convert(BitArray{2}, grey_colony, size(grey_image))

    # sleep(2)
    # imshow(black_image)

    black_poligon = convert(Area{Bool},  black_image)
    black_colony  = grow!(Colony(model, Bool), black_poligon, size(black_image))

    return black_colony
end
