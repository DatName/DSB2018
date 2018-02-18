function rectangle_cell()
    img = zeros(Bool, 200, 200)
    orig_poligon = DSB2018.Model.Area{Bool}()
    for k = 35:45, j = 35 : 45
        img[k, j] = true
        orig_poligon[CartesianIndex(k, j)] = true
    end

    poligon = convert(DSB2018.Model.Area{Bool}, img)

    mdl = DSB2018.Model.XModel(8, 0.2, true, 0.2, 0.9);
    mdl.base_color = 0.8
    mdl.σ = NaN

    x = CartesianIndex(37, 37)
    @assert img[x]

    c = Cell(x, mdl, DSB2018.Model.Area{Bool}());
    DSB2018.Model.Molecular.grow!(c, poligon)

    imshow(c, size(img))
    imshow(img)

    x = collect(keys(orig_poligon))
    y = collect(keys(c.area))

    @show length(x), length(y)
    return (DSB2018.Errors.intersection_over_union(x, y), c)
end

rectangle_cell()
