struct Cell{T <: Any, X <: Any}
    origin::Coordinate
    model::T
    area::Area{X}
    function Cell(origin::Coordinate,
                  model::T,
                  area::Area{X}) where {T <: Any, X <: Any}
        return new{T, X}(origin, model, area)
    end
end

function imshow(this::Cell{T, X}, sz::Tuple{Int64, Int64}) where {T <: Any, X <: Any}
    imshow(convert(BitArray{2}, this.area, sz))
end

function belongs(this::Cell{T, X},
                cur::Tuple{Coordinate, X},
                nxt::Tuple{Coordinate, X},
                poligon::Area{X})::Bool where {T <: Any, X <: Any}
    throw(ErrorException("`belongs` not implemented by Cell{$T, $X}"))
end

function start(this::Cell{T, X})::Coordinate where {T <: Any, X <: Any}
    return this.origin
end

function next(this::Cell{T, X}, poligon::Area{X}, x::Coordinate)::Union{Void, Coordinate} where {T <: Any, X <: Any}
    for z in poligon
        if belongs(this, x, z)
            return z[1]
        end
    end

    #FIXME: this is tooo slow
    # for y in keys(this.area)
    #     for z in poligon
    #         if belongs(this, y, z)
    #             return z[1]
    #         end
    #     end
    # end

    return nothing
end

function done(this::Cell{T, X}, s::Union{Void, Coordinate})::Bool where {T <: Any, X <: Any}
    return s == nothing
end

function grow!(this::Cell{T, X}, poligon::Area{X})::Cell{T, X} where {T <: Any, X <: Any}
    state = start(this)
    while !done(this, state)
        this.area[state] = true
        delete!(poligon, state)
        state = next(this, poligon, state)
    end

    return this
end

function iscell(this::Cell{T, X})::Bool where {T <: Any, X <: Any}
    throw(ErrorException("`iscell(::Cell{$T})` not defined"))
end
