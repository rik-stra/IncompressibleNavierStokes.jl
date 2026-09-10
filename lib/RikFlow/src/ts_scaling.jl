# Normalization for the time-series layer.
#
# This file replaces `scale.jl`. The three functions moved here unchanged -- `_normalise`,
# `scale_input`, `scale_output` -- and a `Scaling` struct was added on top of them.
#
# Why the struct exists. Before it, the normalization convention was a call-site constant: the
# `normalization` field of an entry in `inputs_example.jld2`, read by each training script and then
# discarded. HIT and Taylor-Green used `:standardise` (divide by sigma, mean NOT removed) and the
# channel used `:normal`, so a numerical `lambda` did not mean the same thing on two testbeds, and
# nothing in a saved model recorded which convention had produced it. Three consequences, all of
# them measurement problems rather than style problems:
#
#   1. Uncentred inputs put a large mean component in the design. With `:standardise` the leading
#      eigenvalues of `X'X` are of order `N*(1+(mean/sigma)^2)` against order `N` for `:normal`, so
#      the convention moves the Gram spectrum -- which is the diagnostic that decides what the
#      ridge penalty is actually doing (`meta_files/plan.md` section 8a, `metrics.md` #23).
#   2. A penalized intercept under an uncentred design shrinks the signal level. With
#      `:standardise` the intercept has to carry that level, and it is shrunk along with
#      everything else.
#   3. `metrics.md` #24's split of the level between coefficients and intercept is
#      convention-dependent by construction.
#
# So the convention travels with the fit, as data, and readers assert it. `:normal` plus an
# unpenalized intercept is paper 4's harmonized convention on all three testbeds; the
# paper-2-faithful settings stay reachable (`:standardise`, `penalize_intercept = true`) because
# G1 has to reproduce an archive that was fitted under them.
#
# Stdlib-only on purpose, like the rest of the `ts_*` layer, so it can be `include`d bare by a test
# or a driver without loading IncompressibleNavierStokes or CUDA.

"""
    SCALING_VERSION

Version of the `Scaling` layout. Bump when a field is added or its meaning changes; readers
compare against it so an old file is rejected loudly rather than misread.
"""
const SCALING_VERSION = 1

"""
    NORMALIZATIONS

The modes `_normalise` accepts.

| mode | `mu` | `sigma` |
|---|---|---|
| `:normal` | `mean(x)` | `std(x) + eps` |
| `:standardise` | `0` -- **not centred** | `std(x) + eps` |
| `:minmax` | midpoint | half-range |
| `:Id` | `0` | `1` |
"""
const NORMALIZATIONS = (:normal, :standardise, :minmax, :Id)

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

!!! note
    Kept bit-identical to the `scale.jl` version it replaces: `std` is uncorrected, `ϵ` is added
    after the `std` and is converted to `eltype(x)` first. `fit_scaling` is the recording wrapper
    around it and is what new code should call.
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

# ---------------------------------------------------------------------------------------------
# the convention, carried as data
# ---------------------------------------------------------------------------------------------

"""
    Scaling

An affine QoI normalization together with the convention that produced it.

`mu` and `sigma` are what `scale_input` and `scale_output` use, so a `Scaling` is a drop-in
replacement for the `(; mu, sigma)` named tuple the training scripts used to pass around. The
remaining fields are the part that was previously lost:

- `mode` -- which of [`NORMALIZATIONS`](@ref) produced `mu` and `sigma`. `:standardise` leaves the
  mean in the data, so this is not cosmetic.
- `eps` -- the `ϵ` added to `sigma`. Negligible on HIT enstrophy magnitudes in `Float32`, but it is
  part of the convention and a reproduction to `rtol = 1e-6` can see it.
- `stats_from` -- the series the statistics were computed on. Every archived fit used `:q`: the
  predictor `q_star` and the target are scaled with `q`'s `mu`/`sigma`, which is correct because
  all three live in q-space and the target is a shifted copy of `q`.
- `shared_in_out` -- whether one object scales both directions. Every archived fit shared them
  (`exp_square_HIT/5_train_LinReg.jl:61` builds `(in_scaling = s, out_scaling = s)`), and paper 4
  keeps that. It is recorded rather than implied so a reader can check it instead of assuming it.
- `penalize_intercept` -- not a property of the normalization, but the other half of the
  convention, and useless if it does not travel with it. `false` is paper 4's choice; the archive
  was fitted with `true`.
- `version` -- [`SCALING_VERSION`](@ref).

Construct with [`fit_scaling`](@ref) from data, or with [`as_scaling`](@ref) when adopting a
legacy named tuple whose convention has to be supplied from outside the file.
"""
struct Scaling{M,S,T}
    mode::Symbol
    mu::M
    sigma::S
    eps::T
    stats_from::Symbol
    shared_in_out::Bool
    penalize_intercept::Bool
    version::Int
end

"""
    fit_scaling(x; normalization = :normal, dims = ndims(x), ϵ = 1e-6,
                stats_from = :q, shared_in_out = true, penalize_intercept = false)

Normalize `x` and return `(scaled, scaling::Scaling)`, recording the convention.

The arithmetic is [`_normalise`](@ref)'s, unchanged. Defaults are paper 4's harmonized convention;
pass `normalization = :standardise, penalize_intercept = true` for the paper-2-faithful path that
G1's reproduction needs.
"""
function fit_scaling(x; normalization = :normal, dims = ndims(x), ϵ = 1e-6,
                     stats_from::Symbol = :q, shared_in_out::Bool = true,
                     penalize_intercept::Bool = false)
    normalization in NORMALIZATIONS ||
        error("unknown normalization $normalization; expected one of $NORMALIZATIONS")
    scaled, nt = _normalise(x; normalization, dims, ϵ)
    s = Scaling(normalization, nt.mu, nt.sigma, convert(eltype(x), ϵ), stats_from,
                shared_in_out, penalize_intercept, SCALING_VERSION)
    return scaled, s
end

"""
    as_scaling(nt; normalization, ϵ = 1e-6, stats_from = :q, shared_in_out = true,
               penalize_intercept = true)

Adopt a legacy `(; mu, sigma)` named tuple -- the form stored in every archived `LinReg.jld2` --
as a `Scaling`.

The convention cannot be recovered from the numbers alone and so must be supplied: it lives in the
matching `parameters.jld2` (`normalization`), and `penalize_intercept = true` is a fact about the
code that produced those files (`5_train_LinReg.jl:72`, where the bias column is inside
`kron(I, inp)` and therefore inside `L2Regularization`). Defaults here are the archive's, not
paper 4's, because that is the only thing this function is for.

A `:standardise` tuple can be told from a `:normal` one when `mu` is exactly zero, and that is
checked; nothing else about the convention is inferable.
"""
function as_scaling(nt; normalization::Symbol, ϵ = 1e-6, stats_from::Symbol = :q,
                    shared_in_out::Bool = true, penalize_intercept::Bool = true)
    normalization in NORMALIZATIONS ||
        error("unknown normalization $normalization; expected one of $NORMALIZATIONS")
    if normalization in (:standardise, :Id) && !all(iszero, nt.mu)
        error("normalization = $normalization implies mu == 0, but the stored mu is $(nt.mu)")
    end
    T = eltype(nt.sigma)
    return Scaling(normalization, nt.mu, nt.sigma, convert(T, ϵ), stats_from, shared_in_out,
                   penalize_intercept, SCALING_VERSION)
end

as_scaling(s::Scaling; kwargs...) = s

"""
    scaling_pair(s::Scaling)

The `(; in_scaling, out_scaling)` named tuple the deployed model expects
(`time_series_methods.jl:75-89,133-157`), built from one `Scaling`.

Input and output share one object. That is deliberate and not an oversight: the regression target
is the level `q^{n+1}` and the input carries `q^{n*}` and the lagged `q`, so all of it is q-space
and one set of statistics is the right choice. `s.shared_in_out` records it so a reader can assert
rather than infer.
"""
function scaling_pair(s::Scaling)
    s.shared_in_out || error("scaling_pair called on a Scaling with shared_in_out = false; \
                              build the pair explicitly from two Scaling objects")
    return (in_scaling = s, out_scaling = s)
end

"""
    assert_convention(s; mode = nothing, penalize_intercept = nothing, stats_from = nothing,
                      shared_in_out = nothing, version = SCALING_VERSION)

Check a `Scaling` against the convention the caller expects, and throw naming the mismatch.

Anything that reads a fitted model calls this. The point is that a cross-testbed statement about
`lambda`, a Gram spectrum (#23) or a coefficient/intercept split (#24) is only meaningful once the
convention behind the numbers is known, so the check belongs at the read, not in a comment.
"""
function assert_convention(s::Scaling; mode = nothing, penalize_intercept = nothing,
                           stats_from = nothing, shared_in_out = nothing,
                           version = SCALING_VERSION)
    version === nothing || s.version == version ||
        error("Scaling version $(s.version) != expected $version")
    mode === nothing || s.mode === mode ||
        error("normalization is $(s.mode), expected $mode")
    penalize_intercept === nothing || s.penalize_intercept == penalize_intercept ||
        error("penalize_intercept is $(s.penalize_intercept), expected $penalize_intercept")
    stats_from === nothing || s.stats_from === stats_from ||
        error("statistics were computed on $(s.stats_from), expected $stats_from")
    shared_in_out === nothing || s.shared_in_out == shared_in_out ||
        error("shared_in_out is $(s.shared_in_out), expected $shared_in_out")
    return s
end

assert_convention(nt::NamedTuple; kwargs...) =
    error("this is a legacy (; mu, sigma) scaling and carries no convention. Wrap it with \
           `as_scaling(nt; normalization = ...)`, taking `normalization` from the matching \
           parameters.jld2, before asserting anything about it.")

"""
    is_centred(s::Scaling)

Whether the convention removes the mean. `true` for `:normal` and `:minmax`, `false` for
`:standardise` and `:Id`.

Load-bearing in two places. The Gram spectrum (#23) is read differently in each case, and under a
centred convention with a shared scaling the target is centred too, so an unpenalized intercept
fitted in scaled space should come out near zero -- which is a free check that the convention
change actually took effect.
"""
is_centred(s::Scaling) = s.mode in (:normal, :minmax)

function Base.show(io::IO, s::Scaling)
    print(io, "Scaling(", s.mode, ", stats_from = ", s.stats_from,
          ", penalize_intercept = ", s.penalize_intercept,
          s.shared_in_out ? ", shared" : ", split", ", v", s.version, ")")
end
