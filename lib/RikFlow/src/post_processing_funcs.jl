function ks_dist(data1::Vector,data2::Vector)
    sort!(data1)
    sort!(data2)
    n1 = length(data1)
    n2 = length(data2)
    if min(n1, n2) == 0
        Error("Data passed to ks_dist must not be empty")
    end
    data_all = cat(data1, data2, dims=1)
    cdf1 = map(x->searchsortedlast(data1, x),data_all) / n1
    cdf2 = map(x->searchsortedlast(data2, x),data_all) / n2
    cddiffs = abs.(cdf1 - cdf2)
    argmaxS = argmax(cddiffs)
    loc_maxS = data_all[argmaxS]
    d_1 = cddiffs[argmaxS]
    return d_1, loc_maxS
end


