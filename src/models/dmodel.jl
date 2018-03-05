using OffsetArrays
using ImageFiltering
using ImageFiltering.Kernel
using ImageSegmentation
using StatsBase
import StatsBase: fit!, predict

mutable struct XModel
    size::Int64
    σ::Float64

    base_color::Float64
    range::StepRange{Int64, Int64}
    ncells::Int64
    function XModel(sz::Number, sg::Number)
        return new(convert(Int64, round(sz)), convert(Float64, sg), NaN, -0:1:0, 0)
    end
end

function fit!(this::XModel, rawimg::Matrix{X}) where {X <: Any}
    img = imfilter(rawimg, ImageFiltering.Kernel.gaussian(this.σ))
    if isnan(this.base_color)
        qmin, qmax = extrema(img)
        this.base_color = qmin + (qmax * 0.99 - qmin) / 6.
    end
    this.range = -this.size:1:this.size
    return img
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

function start!(this::Colony{XModel, Bool},
                poligon::Area{Bool},
                sz::Tuple{Int64, Int64})::Coordinate

    this.model.ncells = 0
    img = convert(Matrix{Float64}, poligon, sz)
    img = imfilter(img, ImageFiltering.Kernel.gaussian(3))

    for (k, v) in poligon
        poligon[k] = img[k] > 0.5 ? true : false
    end

    v = collect(values(poligon))
    x = collect(keys(poligon))

    v0, i0 = findmax(v)
    return x[i0]
end

function start!(this::Colony{XModel, X},
                poligon::Area{X},
                sz::Tuple{Int64, Int64})::Coordinate where {X <: Any}

    img = convert(Matrix{Float64}, poligon, sz)
    img = fit!(this.model, img)

    #replace, not add
    for (k, v) in poligon
        poligon[k] = X == Bool ? img[k] > this.model.base_color : img[k]
    end

    v = collect(values(poligon))
    x = collect(keys(poligon))

    v0, i0 = findmax(v)
    return x[i0]
end

function box_average(this::Vector{X}, n::Int64)::Vector{Float64} where {X <: Any}
    out   = similar(this)
    for k = 1 : length(out)
        r = max(1, k - n) : min(length(this), k + n)
        out[k] = mean(this[r])
    end
    return out
end

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

    m0 = c[ci0]
    th = this.model.base_color

    if length(this.cells) > 5 && (X != Bool)
        if this.model.ncells != length(this.cells)
            mv = mean.(this.cells)
            m_predict = box_average(mv, 2)
            δ = mv - m_predict

            mn = m_predict[end]
            th = mn - std(δ)
            this.model.base_color = th
            this.model.ncells = length(this.cells)
        end
    end

    if m0 < th
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

    abs(cur[1] - nxt.first[1]) > 1 && return false
    abs(cur[2] - nxt.first[2]) > 1 && return false

    return true
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
            input_model::XModel; show::Bool = false) where {X <: Any}

    model = deepcopy(input_model)
    grey_image = make_top_white(deepcopy(input))

    grey_poligon = convert(Area{Float64}, grey_image)
    grey_colony  = grow!(Colony(model, Float64),
                            grey_poligon,
                            size(grey_image),
                            show = show)
    #
    black_poligon = Area{Bool}()
    for c in grey_colony.cells
        [black_poligon[x] = true for x in keys(c.area)]
    end

    black_colony  = grow!(Colony(model, Bool),
                            black_poligon,
                            size(grey_image))

    return grey_colony, black_colony
end
