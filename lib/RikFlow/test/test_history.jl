# V1 -- `build_history` against all five archived copies of the history construction -- plus V2
# (batch vs online identity) and the SC-48 index assertion.
#
# V1 is the reproduction oracle. It is the test that found the `fit_ridge` normal-equation bug in
# the 2026-09-05 session, and running it earlier would have saved a day (`plan.md` §25).
#
# Everything here compares code against code. The five copies are transcribed in `Legacy` and run
# on the same inputs as `build_history`; no assertion is made against any document's claim about
# the index convention, because a synthetic system built with an assumed convention confirms
# whichever convention was assumed (`plan.md` §8a).
#
# Shape conventions differ between the two sides and that is not a divergence:
#   - `Legacy` returns `inputs` as `features × samples` with **no** bias row; `fit_model`
#     (`5_train_LinReg.jl:68`) appends the bias afterwards.
#   - `build_history` returns `X` as `samples × features` with the bias as the **last column**.
# So the comparison is `X[:, 1:end-1]' == inputs` and `Y' == outputs`.

@testsnippet HistFixture begin
    using Random
    # A short synthetic record with the real shape contract: `q_star` has T columns, `q` has T+1
    # because `qoisaver` fires on the initial state (`RikFlow.jl:294`).
    function synth(; nq = 6, T = 60, seed = 20260907)
        rng = Random.MersenneTwister(seed)
        q = randn(rng, Float32, nq, T + 1)
        q_star = randn(rng, Float32, nq, T)
        return q, q_star
    end
end

@testitem "V1 q_star_q reproduces the HIT, channel and time_solvers copies" default_imports = false setup = [TSLayer, Legacy, HistFixture] begin
    using Test
    q, q_star = synth()
    T = size(q_star, 2)
    for h in (1, 2, 5, 10)
        spec = TSLayer.HistorySpec(; h, n_qoi = size(q, 1), hist_var = :q_star_q,
                                   include_predictor = true)
        X, Y, steps = TSLayer.build_history(spec, q_star, q)

        # The scripts are handed three equal-length slices; `dQ` is the level shifted one step, so
        # feeding `build_history` the pair `(q_star[:, 1:T], q[:, 1:T+1])` is the same data.
        inputs, outputs = Legacy.hit(h, q_star, q[:, 1:T], q[:, 2:(T + 1)], :q_star_q;
                                     include_predictor = true)

        @test size(X, 1) == size(inputs, 2)
        @test X[:, 1:(end - 1)]' == inputs      # bit-identical, not approximate
        @test Y' == outputs
        @test all(X[:, TSLayer.bias_column(spec)] .== 1)
        @test steps == collect((h + 1):T)

        # Three names, one implementation today. The assertion is what makes a future divergence
        # visible instead of silent.
        @test Legacy.chan(h, q_star, q[:, 1:T], q[:, 2:(T + 1)], :q_star_q) == (inputs, outputs)
        @test Legacy.solvers(h, q_star, q[:, 1:T], q[:, 2:(T + 1)], :q_star_q) == (inputs, outputs)
    end
end

@testitem "V1 q_star history reproduces the archived copies" default_imports = false setup = [TSLayer, Legacy, HistFixture] begin
    using Test
    q, q_star = synth()
    T = size(q_star, 2)
    for h in (1, 3, 5)
        spec = TSLayer.HistorySpec(; h, n_qoi = size(q, 1), hist_var = :q_star)
        X, Y, _ = TSLayer.build_history(spec, q_star, q)
        inputs, outputs = Legacy.hit(h, q_star, q[:, 1:T], q[:, 2:(T + 1)], :q_star)
        @test X[:, 1:(end - 1)]' == inputs
        @test Y' == outputs
    end
end

@testitem "V1 q-only history is offset by one leading row" default_imports = false setup = [TSLayer, Legacy, HistFixture] begin
    using Test
    # 🔴 A real, small divergence, recorded rather than papered over.
    #
    # With `hist_var = :q` the archived copy can form its first row at step `h`, because it needs
    # only `q^{h-1}..q^{0}`. `build_history` uses one row range for all three `hist_var` values and
    # that range is set by `:q_star_q`, which additionally needs `q_star^{n-h}` and therefore
    # cannot start before `n = h+1`. So `build_history` yields one row fewer and the layout is
    # otherwise identical.
    #
    # This matters only for a `:q` fit, and no reported configuration used one (`hist_var` is
    # `:q_star_q` in every archived result), so the offset is documented and left alone rather than
    # changed under a passing test. Should a `:q` cell ever be fitted, this is the assertion that
    # says which convention it inherited.
    q, q_star = synth()
    T = size(q_star, 2)
    for h in (1, 3, 5)
        spec = TSLayer.HistorySpec(; h, n_qoi = size(q, 1), hist_var = :q)
        X, Y, _ = TSLayer.build_history(spec, q_star, q)
        inputs, outputs = Legacy.hit(h, q_star, q[:, 1:T], q[:, 2:(T + 1)], :q)
        @test size(inputs, 2) == size(X, 1) + 1
        @test X[:, 1:(end - 1)]' == inputs[:, 2:end]
        @test Y' == outputs[:, 2:end]
    end
end

@testitem "V1 the Taylor-Green copy is build_history plus a column mask" default_imports = false setup = [TSLayer, Legacy, HistFixture] begin
    using Test
    # TG's copy is not interchangeable with the other three: it **drops** columns where any
    # predictor QoI is small (P0.6's contiguity failure). The informative statement is not that it
    # differs but that it differs by exactly a mask, so the row-dropping can later become a
    # `block_id` boundary rather than a deletion without changing any fitted number.
    q, q_star = synth(; T = 200)
    T = size(q_star, 2)
    h = 5
    nq = size(q, 1)

    # Force the filter to bite: shrink a scattered set of predictor columns below the threshold.
    q_star = copy(q_star)
    small = 20:17:T
    q_star[:, small] .*= Float32(1e-4)

    _, s = TSLayer.fit_scaling(q_star; normalization = :standardise, penalize_intercept = true)
    q_star_sc = TSLayer.scale_input(q_star, s)
    q_sc = TSLayer.scale_input(q, s)
    scaling = TSLayer.scaling_pair(s)

    spec = TSLayer.HistorySpec(; h, n_qoi = nq, hist_var = :q_star_q)
    X, Y, steps = TSLayer.build_history(spec, q_star_sc, q_sc)
    inputs, outputs = Legacy.tg(h, q_star_sc, q_sc[:, 1:T], q_sc[:, 2:(T + 1)], :q_star_q, scaling)

    # The mask, expressed on `build_history`'s rows: row `r` is step `steps[r]`, and TG keeps it
    # when every unscaled predictor QoI at that step exceeds 0.5e-2.
    keep = [all(abs.(q_star_sc[:, n] .* s.sigma) .> 0.5e-2) for n in steps]
    @test size(inputs, 2) == count(keep)
    @test X[keep, 1:(end - 1)]' == inputs
    @test Y[keep, :]' == outputs
    @test count(keep) < length(keep)          # the filter really did fire

    # 🔴 Two findings this assertion pins down, both of which outlive the test.
    #
    # 1. The threshold is 0.5e-2 in training and 1e-2 in deployment
    #    (`time_series_methods.jl:162`), so rows with a predictor in [0.5e-2, 1e-2) are trained on
    #    and then clamped at run time. The two are a factor 2 apart and neither document says so.
    @test 0.5e-2 != 1e-2
    #
    # 2. The filter recovers the raw predictor as `q_star_scaled * sigma`, which is only correct
    #    when `mu == 0` -- i.e. under `:standardise`, the convention TG used. Under paper 4's
    #    `:normal` the same expression is `q_star - mu` and the filter would be testing the wrong
    #    quantity. Any port of this row-dropping to the harmonized convention has to add `mu` back.
    _, sn = TSLayer.fit_scaling(q_star; normalization = :normal)
    @test !all(iszero, sn.mu)
    @test !(TSLayer.scale_input(q_star, sn) .* sn.sigma ≈ q_star)
    @test TSLayer.scale_input(q_star, s) .* s.sigma ≈ q_star
end

@testitem "V1 the fifth, inline online copy" default_imports = false setup = [TSLayer, Legacy, HistFixture] begin
    using Test
    # `time_series_methods.jl:133-146` builds the regressor a sixth way: from the `q_hist` ring
    # buffer, whose column 1 is the most recent step and whose rows split into a corrected half and
    # a predictor half. If this and `build_history` disagree, a negative offline-vs-online rank
    # correlation is indistinguishable from a genuine Ross-et-al. effect -- a false negative on the
    # project's own headline methodological result (`plan.md` §7).
    q, q_star = synth()
    nq, T = size(q_star)
    h = 5
    spec = TSLayer.HistorySpec(; h, n_qoi = nq, hist_var = :q_star_q)
    X, _, steps = TSLayer.build_history(spec, q_star, q)

    for (r, n) in enumerate(steps)
        # The buffer as the deployed loop holds it just before predicting step `n`: column k is
        # step n-k, corrected QoIs on top, predictors underneath.
        q_hist = vcat(reduce(hcat, [q[:, n - k + 1] for k in 1:h]),
                      reduce(hcat, [q_star[:, n - k] for k in 1:h]))
        row = Legacy.online(q_hist, reshape(q_star[:, n], nq, 1), nq, :q_star_q, true)
        @test vec(row) == X[r, :]
    end
end

@testitem "V2 batch equals the online buffer step by step" default_imports = false setup = [TSLayer, HistFixture] begin
    using Test
    q, q_star = synth()
    nq, T = size(q_star)
    for hist_var in (:q_star_q, :q_star, :q), h in (1, 3, 5)
        spec = TSLayer.HistorySpec(; h, n_qoi = nq, hist_var)
        X, _, steps = TSLayer.build_history(spec, q_star, q)
        buf = TSLayer.HistoryBuffer(spec, Float32)
        # Fill the ring with the `h` steps preceding the first row, oldest first.
        for k in (steps[1] - h):(steps[1] - 1)
            push!(buf, q[:, k + 1], q_star[:, k])
        end
        for (r, n) in enumerate(steps)
            @test TSLayer.inputvec(buf, q_star[:, n]) == X[r, :]
            push!(buf, q[:, n + 1], q_star[:, n])
        end
    end
end

@testitem "SC-48 the starred entry at lag k is the same physical step" default_imports = false setup = [TSLayer, HistFixture] begin
    using Test
    # `plan.md` §8a leaves this open as `DEFERRED-TO-G1` and names the arbiter: print the physical
    # time index of every entry of one regressor row and assert elementwise agreement with the
    # deployment step. Three derivations disagreed, so it is settled here from the code.
    #
    # The one premise, stated rather than hidden: `q[:, m+1]` is the corrected QoI at step `m` and
    # `q_star[:, m]` the uncorrected one, the offset `qoisaver` produces by firing on the initial
    # state (`RikFlow.jl:294`). Given that, the test below reads the index off the arrays by
    # planting each step's index as its value, so it cannot be satisfied by a coincidence.
    nq, T, h = 2, 40, 4
    # value = physical step index, distinct per stream
    q = Float64[1000 * i + (m - 1) for i in 1:nq, m in 1:(T + 1)]        # q[:, m+1] -> step m
    q_star = Float64[2000 * i + m for i in 1:nq, m in 1:T]               # q_star[:, m] -> step m

    spec = TSLayer.HistorySpec(; h, n_qoi = nq, hist_var = :q_star_q)
    X, Y, steps = TSLayer.build_history(spec, q_star, q)
    r, n = 1, steps[1]

    # predictor block: step n
    @test X[r, TSLayer.qstar_columns(spec)] == [2000 * i + n for i in 1:nq]
    # target: corrected QoI at step n
    @test Y[r, :] == [1000 * i + n for i in 1:nq]
    for k in 1:h
        cq = TSLayer.qlag_columns(spec, k)
        @test X[r, cq] == [1000 * i + (n - k) for i in 1:nq]             # corrected at step n-k
        cs = (cq[end] + 1):(cq[end] + nq)
        @test X[r, cs] == [2000 * i + (n - k) for i in 1:nq]             # predictor at step n-k
    end
    # ⇒ same-index pairing. The companion is h·N_Q with first block row
    # {A₁+A_*+B₁, A₂+B₂, …, A_h+B_h}, not the (h+1)·N_Q one-step-older form.
    @test TSLayer.nfeatures(spec) == nq * (2h + 1) + 1
end

@testitem "V1 on the tracked HIT record" default_imports = false setup = [TSLayer, Legacy, TrackedData, HistFixture] begin
    using Test
    rec = TrackedData.hit10()
    if rec === nothing
        @warn "no extracted HIT QoI cache under analysis/data/; V1 ran on synthetic data only"
        @test_skip false
    else
        # Paper 2's HIT training window (`plan.md` §8a quotes train_range = (400, 4000)).
        sl = TrackedData.train_slices(rec, (400, 4000))
        @test size(sl.q_bh, 2) == size(sl.q_star_bh, 2) + 1
        for (h, hist_var) in ((5, :q_star_q), (10, :q_star_q), (5, :q_star))
            spec = TSLayer.HistorySpec(; h, n_qoi = size(rec.q, 1), hist_var)
            X, Y, _ = TSLayer.build_history(spec, sl.q_star_bh, sl.q_bh)
            inputs, outputs = Legacy.hit(h, sl.q_star, sl.q_hist, sl.target, hist_var)
            @test X[:, 1:(end - 1)]' == inputs
            @test Y' == outputs
        end
    end
end
