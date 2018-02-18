using OffsetArrays
using ImageFiltering
using ImageFiltering.Kernel

mutable struct XModel
    size::Int64
    q::Float64
    isblack::Bool
    σ::Float64

    base_color::Float64
    kernel::OffsetArray
    range::StepRange{Int64, Int64}
    function XModel(sz::Int64,
                    base_color_quantile::Float64,
                    isblack::Bool,
                    σ::Float64)
        return new(sz, base_color_quantile, isblack, σ,
                    NaN, ImageFiltering.Kernel.gaussian(σ), -sz:1:sz)
    end
end

function black!(this::XModel)::XModel
    this.base_color = 0.5
    this.σ = NaN
    return this
end

function fit!(this::XModel, m::Matrix{Float64})
    nr, nc = size(m)
    for k = 1 : nr
        pline = m[k, :]

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

    imfilter(img, Kernel.DoG(1.))
    a, b = imgradients(img, KernelFactors.sobel, "replicate")
    m    = Images.magnitude(a, b)
    morphogradient(img)

    #NB:2
    hist_equalised_img = clahe(img, 20, xblocks = 8, yblocks = 8, clip = 3)

    img  = Images.imROF(img, 0.2, 20)

    if !isnan(this.model.σ)
        img = imfilter(img, this.model.kernel)
    end

    lt = (x, y) -> this.model.isblack ? x > y : x < y

    #replace, not add
    for (k, v) in poligon
        poligon[k] = X == Bool ? lt.(img[k], this.model.base_color) : img[k]
    end

    v = collect(values(poligon))
    x = collect(keys(poligon))

    #quantile over original values
    if isnan(this.model.base_color)
        this.model.base_color = quantile(v, this.model.q)
    end

    v0, i0 = this.model.isblack ? findmax(v) : findmin(v)
    if X != Bool
        delete!.(poligon, x[v .< this.model.base_color])
    else
        delete!.(poligon, x[ .!v ])
    end

    return x[i0]
end

function next!(this::Colony{XModel, X},
                poligon::Area{X},
                x::Coordinate)::Union{Void, Coordinate} where {X <: Any}

    if isempty(poligon)
        return nothing
    end

    v = collect(values(poligon))
    x = collect(keys(poligon))

    v0, i0 = this.model.isblack ? findmax(v) : findmin(v)

    if this.model.isblack
        if v0 < this.model.base_color
            return nothing
        else
            return x[i0]
        end
    else
        if v0 > this.model.base_color
            return nothing
        else
            return x[i0]
        end
    end
end

function done!(this::Colony{XModel, X}, poligon::Area{X}, x::Union{Void, Coordinate})::Bool where {X <: Any}
    return x == nothing
end

function belongs(this::Cell{XModel, X},
                cur::Coordinate,
                nxt::Pair{Coordinate, X})::Bool where {X <: Any}

    if distance(cur, nxt.first) > this.model.size
        return false
    end

    base_color = this.model.base_color
    nxt_color = nxt.second

    if this.model.isblack
        return nxt_color >= base_color
    else
        return nxt_color <= base_color
    end
end

function iscell(this::Cell{XModel, X})::Bool where {X <: Any}
    return length(this.area) > this.model.size
end

function grow(this::Matrix{X}, model::XModel; show::Bool = false)::Colony{XModel, Bool} where {X <: Any}
    grey_image   = this
    grey_poligon = convert(Area{Float64}, grey_image)
    grey_colony  = grow!(Colony(model, Float64), grey_poligon, size(grey_image), show = show)

    imshow(grey_colony, size(grey_image))

    black_model   = black!(deepcopy(model))
    black_image   = convert(BitArray{2}, grey_colony, size(grey_image))

    sleep(2)
    imshow(black_image)

    black_poligon = convert(Area{Bool},  black_image)
    black_colony  = grow!(Colony(black_model, Bool), black_poligon, size(black_image))

    return black_colony
end
