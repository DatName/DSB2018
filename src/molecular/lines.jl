struct Line
    k::Float64
    b::Float64
    function Line(x::Coordinate, y::Coordinate)
        x1 = x[1]
        y1 = x[2]

        x2 = y[1]
        y2 = y[2]

        if x2 == x1
            throw(DomainError("x2 == x1"))
        end

        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1

        return new(k, b)
    end
end

struct LineSgment
    x::Coordinate
    y::Coordinate
    line::Line

    function LineSegment(x::Coordinate, y::Coordinate)
        return new(x, y, Line(x, y))
    end
end

function ∩(this::Line, that::Line)::Union{Void, Coordinate}
    this.k == that.k && return nothing

    x = (that.b - this.b) / (this.k - that.k)
    y1 = that.k * x + that.b
    y2 = this.k * x + this.b

    y = (y1 + y2) / 2.0

    xr = round(x)
    yr = round(t)

    return Coordinate(Int64(xr), Int64(yr))
end
