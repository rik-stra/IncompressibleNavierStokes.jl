"""
    _normalise(x; normalization=:normal, dims=ndims(x), ϵ=1e-6)

Normalize the input array `x` using the specified normalization method.

# Arguments
- `x`: The input array to be normalized.
- `normalization`: The normalization method to use. Can be `:normal` for standard normalization, `:minmax` for min-max normalization, `:standardise` for standardization. Default is `:normal`.
- `dims`: The dimensions over which to compute the normalization statistics. Default is `ndims(x)`.
- `ϵ`: A small value added to the standard deviation to avoid division by zero. Default is `1e-6`.

# Returns
- A tuple containing the normalized array and a named tuple with the computed mean (`mu`) and standard deviation (`sigma`).

# Normalization Methods
- `:normal`: Standard normalization where the mean (`mu`) and standard deviation (`sigma`) are computed along the specified dimensions.
- `:minmax`: Min-max normalization where the minimum and maximum values are computed along the specified dimensions, and the mean (`mu`) is set to the midpoint and the standard deviation (`sigma`) to half the range.
- `:standardise`: Standardization where the mean (`mu`) is set to zero and the standard deviation (`sigma`) is standardized along the specified dimensions.
"""
function _normalise(x; normalization= :normal, dims=ndims(x), ϵ=1e-6)
    if normalization == :normal
        ϵ = convert(eltype(x), ϵ)
        mu = mean(x, dims=dims)
        sigma = std(x, dims=dims, corrected=false).+ ϵ
        
    elseif normalization == :minmax
        min = minimum(x, dims=dims)
        max = maximum(x, dims=dims)
        mu = convert(eltype(x), 0.5)*(min+max)
        sigma = convert(eltype(x), 0.5)*(max-min)
    elseif normalization == :standardise
        ϵ = convert(eltype(x), ϵ)
        sigma = std(x, dims=dims, corrected=false).+ ϵ
        mu = convert(eltype(x),0)
    elseif normalization == :Id
        mu = convert(eltype(x),0)
        sigma = convert(eltype(x),1)
    end
    return (x .- mu) ./ (sigma), (;mu,sigma)
end

function scale_input(x::AbstractArray, scaling)
    return (x .- scaling.mu) ./ scaling.sigma
end

function scale_output(x::AbstractArray, scaling)
    return x .* scaling.sigma .+ scaling.mu
end