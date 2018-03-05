import Base: mean

struct Colony{T <: Any, X <: Any}
    model::T
    cells::Vector{Cell{T, X}}
    function Colony(model::T, ::Type{X}) where {T <: Any, X <: Any}
        return new{T, X}(model, Cell{T, X}[])
    end
end

function mean(this::Colony{T, X})::X where {T <: Any, X <: Any}
    return mean(mean.(this.cells))
end

function convert(::Type{BitArray{2}},
                this::Colony{T, X},
                sz::Tuple{Int64, Int64}) where {T <: Any, X <: Any}
    out = BitArray{2}(sz[1], sz[2]) .* false
    for cell in this.cells
        for k in keys(cell.area)
            out[k] = true
        end
    end

    return out
end

function get_random_color(seed)
    srand(seed)
    rand(RGB{Float64})
end

function convert(::Type{Matrix{RGB{Float64}}},
                this::Colony{T, X},
                sz::Tuple{Int64, Int64}) where {T <: Any, X <: Any}

    out = Matrix{RGB{Float64}}(sz)
    for x in CartesianRange(CartesianIndex(1, 1), CartesianIndex(sz))
        out[x] = RGB(0.0, 0.0, 0.0)
    end

    c = 0
    for cell in this.cells
        color = get_random_color(c)
        for k in keys(cell.area)
            out[k] = color
        end
        c += 1;
    end

    return out
end

function imshow(this::Colony{T, X}, sz::Tuple{Int64, Int64}) where {T <: Any, X <: Any}
    imshow(convert(Matrix{RGB{Float64}}, this, sz))
end

"""Can mutate both Colony and poligon"""
function start!(this::Colony{T, X}, poligon::Area{X}, sz::Tuple{Int64, Int64})::Coordinate where {T <: Any, X <: Any}
    throw(ErrorException("`start(::Colony{$T}, ::Area{X})` not implemented"))
end

"""Can mutate both Colony and poligon"""
function next!(this::Colony{T, X}, poligon::Area{X}, x::Coordinate)::Union{Void, Coordinate} where {T <: Any, X <: Any}
    throw(ErrorException("`next(::Colony{$T}, ::Area{$X}, ::Coordinate)` not implemented"))
end

"""Can mutate both Colony and poligon"""
function done!(this::Colony{T, X}, poligon::Area{X}, x::Union{Void, Coordinate}) where {T <: Any, X <: Any}
    throw(ErrorException("`done(::Colony{$T}, ::Area{$X}, ::Union{Void, Coordinate})` not implemented"))
end

function grow!(this::Colony{T, X},
                poligon::Area{X},
                sz::Tuple{Int64, Int64};
                show::Bool = false)::Colony{T, X} where {T <: Any, X <: Any}

    empty_cells = 0
    x = start!(this, poligon, sz)
    while true
        println("Collecting cell from ", x)

        cell = Cell(x, this.model, Area{X}())
        cell = grow!(cell, poligon)

        @show length(cell.area)

        if iscell(cell)
            push!(this.cells, cell)
            if show
                imshow(this, sz)
                sleep(5)
            end
        else
            empty_cells += 1
        end

        if length(this.cells) > 2
            break
        end

        x = next!(this, poligon, sz)
        done!(this, poligon, x) && break
    end

    @show empty_cells
    return this
end

function iscolony(this::Colony{T, X})::Bool where {T <: Any, X <: Any}
    return length(this.cells) > 0
end
