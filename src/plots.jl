function getColoredPredictionImage(this::DSB2018.Model.ThickSet)
    n, r = size(this.poligon)
    out = zeros(RGB{Float64}, n, r)
    for line in this.lines
        @show line.intensity
        c = rand(RGB{Float64})
        for x in keys(line.area)
            if out[x] != RGB(0., 0., 0.)
                @show x
                throw("A")
            end
            out[x] = c
        end
    end

    return out
end
