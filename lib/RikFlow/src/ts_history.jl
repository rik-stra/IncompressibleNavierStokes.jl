# Shared history builder for the TO time-series models.
#
# One definition, used by both the offline fit and the online rollout. The four script-local
# copies of `create_history` (exp_square_HIT/5_train_LinReg.jl:22, channel/5_train_LinReg.jl:22,
# taylor-green/7_train_LinReg.jl:23, time_solvers/train_linreg.jl:18) plus the inline construction
# in time_series_methods.jl:134-146 are all to be replaced by this.
#
# TODO(V1): bit-identity test against all four archived copies. Not yet done.

"""
    HistorySpec(; h, n_qoi, hist_var = :q_star_q, include_predictor = true)

Specification of the lagged-history regressor.

# Arguments
- `h`: history length (number of lags).
- `n_qoi`: number of QoIs, `N_Q`.
- `hist_var`: which streams enter the history. `:q` (corrected only), `:q_star` (predictor only),
  or `:q_star_q` (both, interleaved per lag).
- `include_predictor`: whether the current predictor `q^{n*}` is prepended.

# Layout
With `hist_var = :q_star_q` and `include_predictor = true` the row is

    [ q^{n*} | q^{n-1}, q^{n-1*} | ... | q^{n-h}, q^{n-h*} | 1 ]

of length `N_Q(2h+1) + 1`, matching `methodology.md` §0 item 5. The bias column is always last.
"""
Base.@kwdef struct HistorySpec
    h::Int
    n_qoi::Int
    hist_var::Symbol = :q_star_q
    include_predictor::Bool = true
end

"""
    nfeatures(spec::HistorySpec)

Number of columns of the regressor, including the trailing bias column.
"""
function nfeatures(spec::HistorySpec)
    nq = spec.n_qoi
    nstream = spec.hist_var == :q_star_q ? 2 : 1
    return nq * nstream * spec.h + (spec.include_predictor ? nq : 0) + 1
end

"""
    build_history(spec, q_star, q)

Build the batch regressor and target from a tracked run.

# Arguments
- `q_star`: `N_Q × T`, the predictor QoIs. Column `n` is step `n`.
- `q`: `N_Q × (T+1)`, the corrected QoIs. **Column 1 is t = 0**, so `q[:, n+1]` is the corrected
  QoI at step `n`. This offset is inherited from `qoisaver` firing on the initial state
  (`RikFlow.jl:294`) and the training scripts rely on it (`7_train_LinReg.jl:64-66`).

# Returns
`(X, Y, steps)` with `X` of size `(N, nfeatures(spec))`, `Y` of size `(N, N_Q)`, and `steps` the
physical step index of each row. Rows are ordered by increasing step, which is what the
autoregressive residual shift in `ts_fit.jl` requires.
"""
function build_history(spec::HistorySpec, q_star, q)
    nq, h = spec.n_qoi, spec.h
    T = size(q_star, 2)
    @assert size(q, 2) == T + 1 "q must have one more column than q_star (t=0 included)"
    @assert size(q, 1) == nq && size(q_star, 1) == nq

    steps = collect((h + 1):T)
    N = length(steps)
    m = nfeatures(spec)
    Tel = promote_type(eltype(q), eltype(q_star))
    X = zeros(Tel, N, m)
    Y = zeros(Tel, N, nq)

    for (r, n) in enumerate(steps)
        col = 1
        if spec.include_predictor
            X[r, col:(col + nq - 1)] .= q_star[:, n]
            col += nq
        end
        for k in 1:h
            if spec.hist_var == :q || spec.hist_var == :q_star_q
                # q^{n-k} is the corrected QoI at step n-k, i.e. column n-k+1 of q
                X[r, col:(col + nq - 1)] .= q[:, n - k + 1]
                col += nq
            end
            if spec.hist_var == :q_star || spec.hist_var == :q_star_q
                X[r, col:(col + nq - 1)] .= q_star[:, n - k]
                col += nq
            end
        end
        X[r, col] = one(Tel)
        Y[r, :] .= q[:, n + 1]
    end
    return X, Y, steps
end

"""
    HistoryBuffer(spec, Tel)

Online counterpart of [`build_history`](@ref): a ring of the last `h` corrected and predicted QoI
vectors, from which a single regressor row is formed each step.

`push!(buf, q, q_star)` records one completed step; `inputvec(buf, q_star)` forms the row for the
step about to be taken. The batch-vs-online identity between this and `build_history` is V2, the
hard prerequisite for the calibration gate.
"""
struct HistoryBuffer{Tel}
    spec::HistorySpec
    q::Matrix{Tel}       # n_qoi × h, column 1 is the most recent
    q_star::Matrix{Tel}  # n_qoi × h, column 1 is the most recent
    nfilled::Base.RefValue{Int}
end

function HistoryBuffer(spec::HistorySpec, ::Type{Tel}) where {Tel}
    HistoryBuffer{Tel}(
        spec,
        zeros(Tel, spec.n_qoi, spec.h),
        zeros(Tel, spec.n_qoi, spec.h),
        Ref(0),
    )
end

function Base.push!(buf::HistoryBuffer, q, q_star)
    h = buf.spec.h
    h == 0 && return buf
    for j in h:-1:2
        buf.q[:, j] .= buf.q[:, j - 1]
        buf.q_star[:, j] .= buf.q_star[:, j - 1]
    end
    buf.q[:, 1] .= q
    buf.q_star[:, 1] .= q_star
    buf.nfilled[] = min(buf.nfilled[] + 1, h)
    return buf
end

"""
    inputvec(buf::HistoryBuffer, q_star)

Form the regressor row for the current step, given the current predictor `q_star`.
"""
function inputvec(buf::HistoryBuffer{Tel}, q_star) where {Tel}
    spec = buf.spec
    nq, h = spec.n_qoi, spec.h
    x = zeros(Tel, nfeatures(spec))
    col = 1
    if spec.include_predictor
        x[col:(col + nq - 1)] .= q_star
        col += nq
    end
    for k in 1:h
        if spec.hist_var == :q || spec.hist_var == :q_star_q
            x[col:(col + nq - 1)] .= buf.q[:, k]
            col += nq
        end
        if spec.hist_var == :q_star || spec.hist_var == :q_star_q
            x[col:(col + nq - 1)] .= buf.q_star[:, k]
            col += nq
        end
    end
    x[col] = one(Tel)
    return x
end

"""
    qstar_columns(spec)

Column range of the current predictor `q^{n*}` in a regressor row, or `nothing` when the spec does
not include it.
"""
qstar_columns(spec::HistorySpec) =
    spec.include_predictor ? (1:spec.n_qoi) : nothing

"""
    qlag_columns(spec, k)

Column range of the corrected QoI at lag `k`, `q^{n-k}`, or `nothing` when the spec carries no
corrected-QoI history. These are the only columns of `C` that close a feedback loop during a
free-running rollout: everything else is either exogenous (the predictor stream) or constant.
"""
function qlag_columns(spec::HistorySpec, k::Int)
    (1 <= k <= spec.h) || return nothing
    spec.hist_var == :q_star && return nothing
    nq = spec.n_qoi
    stride = spec.hist_var == :q_star_q ? 2nq : nq
    start = (spec.include_predictor ? nq : 0) + (k - 1) * stride + 1
    return start:(start + nq - 1)
end

"""
    qstarlag_columns(spec, k)

Column range of the **predictor** at lag `k`, `q^{n-k*}`, or `nothing` when the spec carries no
predictor history.

These columns are exogenous during a free-running rollout -- the solver supplies them -- which is
why they need a closure before any companion propagator exists (`ts_spectrum.jl`). The pairing is
same-index: this range and `qlag_columns(spec, k)` refer to the **same physical step**, settled
from the code by the SC-48 assertion in `test/test_history.jl`, not from either document's claim.
"""
function qstarlag_columns(spec::HistorySpec, k::Int)
    (1 <= k <= spec.h) || return nothing
    spec.hist_var === :q && return nothing
    nq = spec.n_qoi
    base = (spec.include_predictor ? nq : 0)
    if spec.hist_var === :q_star_q
        start = base + (k - 1) * 2nq + nq + 1     # after that lag's corrected block
    else                                          # :q_star -- the predictor stream alone
        start = base + (k - 1) * nq + 1
    end
    return start:(start + nq - 1)
end

"""
    bias_column(spec)

Index of the trailing bias column.
"""
bias_column(spec::HistorySpec) = nfeatures(spec)
