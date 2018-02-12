mutable struct ThickLine
    start::Coordinate
    intensity::Float64 #vertical tolerace
    sharpness::Float64 #horizontal tolerance
    area::Area
    poligon::Matrix{Float64}
    exclude::Area
    function ThickLine(coord_start::Coordinate,
                        intensity::Float64,
                        sharpness::Float64,
                        poligon::Matrix{Float64};
                        exclude::Area = Area())
        return new(coord_start, intensity, sharpness, Area(), poligon, exclude)
    end
end

function issimilar(this::ThickLine, x::Coordinate, y::Coordinate)::Bool
    return this.poligon[x] > this.intensity
end

function start(this::ThickLine)::Coordinate
    return this.start
end

function next(this::ThickLine, point::Coordinate)::Tuple{Coordinate, Float64}
    n = size(this.poligon, 1)
    m = size(this.poligon, 2)

    for x in neighbors(point, n, m, r = Int64(ceil(this.sharpness)))
        if (x ∈ this.area) || (x ∈ this.exclude)
            continue
        end

        if issimilar(this, x, point)
            return x, this.poligon[x]
        end
    end

    for z in keys(this.area)
        for x in neighbors(z, n, m, r = Int64(ceil(this.sharpness)))
            if (x ∈ this.area) || (x ∈ this.exclude)
                continue
            end

            if issimilar(this, x, point)
                return x, this.poligon[x]
            end
        end
    end

    return CartesianIndex{2}(0, 0), NaN
end

function done(this::ThickLine, c::Coordinate)::Bool
    return (c[1] == 0) || (c[2] == 0)
end

function collect(this::ThickLine)::Vector{Coordinate}
    state = start(this)
    while !done(this, state)
        this.area[state] = this.poligon[state]
        state, val = next(this, state)
    end
    return collect(keys(this.area))
end

function convert(::Type{Matrix{Float64}}, this::ThickLine)::Matrix{Float64}
    m = zeros(Float64, size(this.poligon))*0.0;
    for (k, v) in this.area
        m[k] = v
    end
    return m
end

function convert(::Type{Matrix{Bool}}, this::ThickLine)::Matrix{Bool}
    m = falses(size(this.poligon));
    for (k, v) in this.area
        m[k] = true
    end
    return convert(Matrix{Bool}, m)
end

function isborder(this::Vector{Coordinate},
                    x::Coordinate,
                    numrows::Int64,
                    numcols::Int64;
                    r::Int64 = 1)

    any_out = false
    any_in  = false

    for z in neighbors(x, numrows, numcols, r = r)
        if !any_in && (z ∈ this)
            any_in = true
        end

        if !any_out && (z ∉ this)
            any_out = true
        end

        any_in && any_out && break
    end

    return any_out && any_in
end

function getBorder(this::ThickLine)::Vector{Coordinate}
    out = Coordinate[]
    numrows, numcols = size(this.poligon)
    l = collect(keys(this.area))
    for x in l
        isborder(l, x, numrows, numcols) && push!(out, x)
    end

    return out
end

using ImageView
import ImageView: imshow

function imshow(this::Vector{Coordinate}, sz::Tuple{Int64, Int64})
    out = falses(sz)
    for x in this
        out[x] = true
    end

    imshow(out)
end

function imshow(this::ThickLine)
    l = collect(keys(this.area))
    imshow(l, size(this.poligon))
end
