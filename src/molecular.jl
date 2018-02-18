module Molecular

import Base: done, convert, ∈, ∩, ∪
import ImageView: imshow

using ColorTypes

export Cell, Colony, Line, LineSegment
export belongs, start!, next!, iscell
export ∈, grow!, grow, ∩, ∪
export convert, imshow

const Coordinate = CartesianIndex{2}
const Area{T} = Dict{Coordinate, T}

function ∈(x::Coordinate, area::Area{T})::Bool where {T <: Any}
    return haskey(area, x)
end

include("./molecular/lines.jl")
include("./molecular/cell.jl")
include("./molecular/colony.jl")

end
