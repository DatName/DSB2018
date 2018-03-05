module Molecular

import Base: done, convert, ∈, ∩, ∪
import ImageView: imshow
import Base: getindex
using DSB2018.Model
using ColorTypes

export Cell, Colony, Line, LineSegment
export belongs, start!, next!, iscell
export ∈, grow!, grow, ∩, ∪
export convert, imshow, getindex

const Coordinate = CartesianIndex{2}
const Area{T} = Dict{Coordinate, T}

function ∈(x::Coordinate, area::Area{T})::Bool where {T <: Any}
    return haskey(area, x)
end

include("./molecular/cell.jl")
include("./molecular/colony.jl")

end
