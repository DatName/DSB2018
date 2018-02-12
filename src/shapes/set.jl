using ColorTypes

struct ThickSet
    lines::Vector{ThickLine}
    covered::Area
    poligon::Matrix{Float64}
    avg_size::Int64
    sharpness::Float64
    function ThickSet(poligon::T,
                        avg_size::Int64,
                        sharpness::Float64) where {T <: Any}
        return new(ThickLine[],
                    Area(),
                    convert(Matrix{Float64}, poligon),
                    avg_size,
                    sharpness)
    end
end

function start(this::ThickSet)::Coordinate
    m = -Inf
    idx = CartesianIndex{2}(-1, -1)
    for x in CartesianRange(CartesianIndex{2}(1, 1), CartesianIndex{2}(size(this.poligon)))
        if x ∈ this.covered
            continue
        end

        v = this.poligon[x]
        if v > m
            m = v
            idx = x
        end
    end

    return idx
end

function next(this::ThickSet, j::Coordinate)::Tuple{ThickLine, Coordinate}
    n, m = size(this.poligon)
    avg_size = this.avg_size
    intensity = 0.0
    nc = 0.0

    for x in neighbors(j, n, m, r = Int64(ceil(avg_size)))
        if x ∈ this.covered
            continue
        end

        intensity += this.poligon[x]
        nc += 1.0
    end

    next_value = -Inf
    next_index = CartesianIndex{2}(0, 0)

    break_trigger = (ThickLine(next_index, NaN, NaN, this.poligon), next_index)

    if nc == 0
        return break_trigger
    end

    intensity = intensity / nc
    #NB: break condition on area intensity is bad
    if intensity < this.sharpness
        return break_trigger
    end

    line = ThickLine(j, intensity, this.sharpness, this.poligon, exclude = this.covered)
    collect(line)

    merge!(this.covered, line.area)

    for x in CartesianRange(CartesianIndex{2}(1, 1), CartesianIndex{2}(n, m))
        if (x ∈ this.covered)
            continue
        end

        if this.poligon[x] > next_value
            next_value = this.poligon[x]
            next_index = x
        end
    end

    return (line, next_index)
end

function done(this::ThickSet, x::Coordinate)::Bool
    return x[1] <= 0 || x[2] <= 0
end

#NB: break condition on area length is bad
function collect(this::ThickSet)
    state = start(this)
    while !done(this, state)
        line, state = next(this, state)
        if isnan(line.intensity) || isnan(line.sharpness)
            break
        end

        if length(line.area) < this.avg_size / 2
            break
        end

        push!(this.lines, line)
        @printf("Line #%d ready [%d]\n", length(this.lines), length(line.area))
    end

    return this
end

function convert(::Type{Matrix{RGB{Float64}}}, this::ThickSet)
    m = zeros(RGB{Float64}, size(this.poligon))

    for l in this.lines
        c = RGB(rand(), rand(), rand())
        for k in keys(l.area)
            m[k] = c
        end
    end

    return m
end

function convert(::Type{Matrix{Bool}}, this::ThickSet)
    m = falses(size(this.poligon))

    for l in this.lines
        for (k, v) in l.area
            m[k] = true
        end
    end

    return m
end

function imshow(this::ThickSet)
    c = convert(Matrix{RGB{Float64}}, this)
    imshow(c)
end
