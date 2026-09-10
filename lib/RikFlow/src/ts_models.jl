# Model container for the conditional-density time-series ladder.
#
# One struct covers every rung; the ladder is nesting by zeroing blocks:
#   ar_order = 0                     -> white residual
#   W zero outside its bias row      -> constant Sigma
#   C zero outside its bias row      -> the data-driven noise model of paper 1
#
# Sigma^n is the covariance of the INNOVATION xi, never of the marginal eta. See the plan's D2:
# imposing a constant *marginal* correlation can require an innovation covariance that is not PSD.

"""
    pacf_to_ar(phi)

Map partial autocorrelations `phi` (each in `(-1, 1)`) to AR coefficients by the Levinson-Durbin
recursion. The image is exactly the stationarity region, so any `phi` in the open cube gives a
stationary AR(p) and no constraint has to be policed during optimisation
(Monahan / Barndorff-Nielsen-Schou).
"""
function pacf_to_ar(phi::AbstractVector{T}) where {T}
    p = length(phi)
    a = zeros(T, p)
    for k in 1:p
        prev = copy(a)
        a[k] = phi[k]
        for j in 1:(k - 1)
            a[j] = prev[j] - phi[k] * prev[k - j]
        end
    end
    return a
end

"""
    ar_roots(a)

Roots of the AR characteristic polynomial `1 - a_1 z - ... - a_p z^p`, returned as the reciprocals
`z^{-1}` so that stationarity is `abs(root) < 1`. Used to report the decorrelation time `T^eta` and,
where the pair is complex, the oscillation frequency.
"""
function ar_roots(a::AbstractVector)
    p = length(p_nonzero(a))
    p == 0 && return ComplexF64[]
    # companion matrix of the AR recursion
    F = zeros(Float64, p, p)
    F[1, :] .= Float64.(a[1:p])
    for j in 2:p
        F[j, j - 1] = 1.0
    end
    return eigvals(F)
end

p_nonzero(a) = a  # kept explicit: order is fixed by ar_order, not by trailing zeros

"""
    decorrelation_time(a, dt)

Decorrelation time `T^eta` implied by the AR coefficients, from the slowest root:
`T^eta = -dt / log|z_max|`. Returns `(T_eta, omega)` with `omega` the angular frequency of the
slowest root (zero for a real root).

`T^eta` is deliberately **not** written `tau`: `tau_i(t)` is already the TO coefficient of QoI `i`
in paper 2's ansatz `m = sum_i tau_i(t) O_i(v)`, stored in the code as `tau = dQ ./ src_Q`.
"""
function decorrelation_time(a::AbstractVector, dt)
    r = ar_roots(a)
    isempty(r) && return (zero(dt), zero(dt))
    i = argmax(abs.(r))
    z = r[i]
    m = abs(z)
    m >= 1 && return (typemax(Float64), angle(z) / dt)
    return (-dt / log(m), abs(angle(z)) / dt)
end

"""
    JointModel

Fitted conditional-density model.

# Fields
- `spec`: the [`HistorySpec`](@ref) whose layout `C` and `W` are written against.
- `C`: `m x N_Q` mean regression matrix (features x QoIs).
- `W`: `m x N_Q` log-scale head. The bias row carries the unconditional log scale, so `W` zero
  outside it is a constant `Sigma`.
- `R`: `N_Q x N_Q` constant correlation matrix of the innovation.
- `psi`: `p x N_Q` pre-activation AR parameters; `phi = tanh(psi)` are the partial autocorrelations.
- `uclip`: `(u_min, u_max)` bounds on the log-scale pre-activation, or `nothing`.
- `scaling`: the `(mu, sigma)` used to standardise inputs and outputs.
"""
struct JointModel{T}
    spec::HistorySpec
    C::Matrix{T}
    W::Matrix{T}
    R::Matrix{T}
    psi::Matrix{T}
    uclip::Union{Nothing,Tuple{T,T}}
    scaling::NamedTuple
end

ar_order(m::JointModel) = size(m.psi, 1)
n_qoi(m::JointModel) = size(m.C, 2)

"""
    ar_coefficients(m::JointModel)

`p x N_Q` matrix of AR coefficients, column `i` being the diagonal entries `a_{k,i}` of `A_k`.
"""
function ar_coefficients(m::JointModel{T}) where {T}
    p, nq = size(m.psi)
    a = zeros(T, p, nq)
    for i in 1:nq
        a[:, i] .= pacf_to_ar(tanh.(m.psi[:, i]))
    end
    return a
end

"""
    nparams(m::JointModel)

Free parameter count, counting `R` as concentrated out of the likelihood (it is recomputed in
closed form from the standardised residuals rather than optimised).
"""
nparams(m::JointModel) = length(m.C) + length(m.W) + length(m.psi)
