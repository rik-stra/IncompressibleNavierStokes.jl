# V17, V26, V27 -- the metric layer.
#
# Every metric here is checked against something that is not itself: a closed form, a BigFloat
# reference computed in the test, a Monte-Carlo estimate, or a synthetic process with a known
# answer. None is checked against a number quoted from a document.

@testsnippet ScoreFixture begin
    using Random, LinearAlgebra, Statistics

    "erf by its Maclaurin series in BigFloat -- the reference, computed here, not remembered."
    function bigerf(x)
        z = BigFloat(x)
        s = z
        t = z
        z2 = z * z
        for n in 1:2000
            t *= -z2 / n
            a = t / (2n + 1)
            s += a
            abs(a) <= eps(BigFloat) * abs(s) && break
        end
        return 2 / sqrt(BigFloat(pi)) * s
    end

    """
        exchangeable_ar1(N, M; a, rng)

    The null for V27: `M+1` independent AR(1) paths with the same parameter, one designated truth.

    At every step the `M+1` values are exchangeable, so the ranks are marginally uniform -- and
    because each path is smooth in time the **rank series is serially correlated**, which is the
    dependence the `N_eff` correction exists to handle. Drawing the members i.i.d. per step would
    give uniform *and independent* ranks and the test would have nothing to detect.
    """
    function exchangeable_ar1(N, M; a = 0.94, rng = Random.default_rng())
        paths = zeros(N, M + 1)
        s = sqrt(1 - a^2)
        for m in 1:(M + 1)
            x = randn(rng)
            for n in 1:N
                x = a * x + s * randn(rng)
                paths[n, m] = x
            end
        end
        return (ens = paths[:, 1:M], truth = paths[:, M + 1])
    end
end

@testitem "V17 erf and the normal CDF against a BigFloat reference" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test
    for x in (0.0, 1e-8, 0.1, 0.5, 1.0, 1.9, 2.0, 2.0001, 2.5, 3.0, 4.0, 6.0)
        ref = Float64(bigerf(x))
        @test TSLayer.erf_(x) ≈ ref rtol = 1e-14
        @test TSLayer.erf_(-x) ≈ -ref rtol = 1e-14
        @test TSLayer.erfc_(x) ≈ 1 - ref atol = 1e-16 rtol = 1e-12
    end
    # The switch at |x| = 2 must not leave a step.
    @test TSLayer.erf_(2 - 1e-12) ≈ TSLayer.erf_(2 + 1e-12) rtol = 1e-13
    @test TSLayer.norm_cdf(0.0) == 0.5
    @test TSLayer.norm_cdf(1.0) ≈ 0.8413447460685429 rtol = 1e-14
    @test TSLayer.norm_cdf(-1.0) ≈ 0.15865525393145707 rtol = 1e-13
    # A far tail keeps its relative accuracy instead of cancelling against 1.
    @test TSLayer.norm_cdf(-6.0) ≈ 9.865876450376946e-10 rtol = 1e-10
    @test TSLayer.norm_cdf(-6.0) > 0
    @test TSLayer.norm_pdf(0.0) ≈ 1 / sqrt(2pi) rtol = 1e-15
end

@testitem "V17 NLL against a hand-computed Gaussian" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test, LinearAlgebra, Random
    rng = Random.MersenneTwister(1)
    # One QoI, unit variance, so the per-sample NLL is 0.5(z^2 + log 2pi) averaged.
    Y = reshape([1.0, -0.5, 2.0], 3, 1)
    got = TSLayer.nll_gaussian(Y, [0.0], reshape([1.0], 1, 1))
    want = sum(0.5 * (y^2 + log(2pi)) for y in Y)
    @test got.total ≈ want
    @test got.per_sample ≈ want / 3
    @test got.per_sample_per_qoi ≈ want / 3

    # A correlated bivariate case, against the explicit formula.
    S = [2.0 0.6; 0.6 1.0]
    Y2 = randn(rng, 50, 2)
    mu = randn(rng, 2)
    byhand(Y2, mu, S) = sum(0.5 * (logdet(S) + (Y2[n, :] .- mu)' * (S \ (Y2[n, :] .- mu)) +
                                   size(Y2, 2) * log(2pi)) for n in axes(Y2, 1))
    tot = byhand(Y2, mu, S)
    @test TSLayer.nll_gaussian(Y2, mu, S).total ≈ tot

    # A per-step Sigma reproduces the shared-Sigma answer when every step is the same.
    @test TSLayer.nll_gaussian(Y2, mu, [S for _ in 1:50]).total ≈ tot

    # The minimum sits where the width matches the error: scaling Sigma away from truth costs.
    Y3 = randn(rng, 4000, 1)
    at1 = TSLayer.nll_gaussian(Y3, [0.0], reshape([1.0], 1, 1)).per_sample
    @test at1 < TSLayer.nll_gaussian(Y3, [0.0], reshape([0.25], 1, 1)).per_sample
    @test at1 < TSLayer.nll_gaussian(Y3, [0.0], reshape([4.0], 1, 1)).per_sample
end

@testitem "V17 CRPS closed form against Monte Carlo, and the fair form against the biased one" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test, Random, Statistics
    rng = Random.MersenneTwister(20260907)

    # Closed form vs a large fair-ensemble estimate from the same Gaussian.
    for (mu, sigma, y) in ((0.0, 1.0, 0.5), (2.0, 0.5, 2.0), (-1.0, 3.0, 4.0), (0.0, 1.0, -2.5))
        closed = TSLayer.crps_gaussian(mu, sigma, y)
        x = mu .+ sigma .* randn(rng, 200_000)
        @test TSLayer.crps_ensemble(x, y) ≈ closed rtol = 0.02
    end
    # At zero width it is the absolute error.
    @test TSLayer.crps_gaussian(1.0, 0.0, 3.0) ≈ 2.0
    @test TSLayer.crps_gaussian(1.0, 1e-12, 3.0) ≈ 2.0 atol = 1e-6

    # Propriety, the property that makes it usable: the score is minimised by the true width.
    ys = randn(rng, 20_000)
    score(s) = sum(TSLayer.crps_gaussian(0.0, s, y) for y in ys) / length(ys)
    @test score(1.0) < score(0.5)
    @test score(1.0) < score(2.0)

    # 🔑 The fair denominator is not a detail at M = 5, the archive's ensemble size. The biased
    # 1/(2M^2) form subtracts too little spread, so it scores an ensemble as better than it is --
    # and the error is largest exactly where the spread term matters, which is under-dispersion.
    M = 5
    x = randn(rng, M)
    fair = TSLayer.crps_ensemble(x, 0.3; fair = true)
    biased = TSLayer.crps_ensemble(x, 0.3; fair = false)
    @test biased > fair                       # biased looks *worse* here because it subtracts less
    spread_term = biased - fair
    @test spread_term > 0
    # The two denominators differ by exactly a factor M/(M-1) on the spread term: 25% at M = 5,
    # which is recoverable from the two scores because the first term is common to both.
    g = 2 * M * M * (biased - fair) / (M / (M - 1) - 1)      # the raw double sum
    @test fair ≈ biased - g / (2 * M * M) * (M / (M - 1) - 1) rtol = 1e-12

    # And the fair form is the unbiased one: over many draws from the truth's own law, the fair
    # ensemble CRPS converges to the closed form while the biased one does not.
    function mean_scores(nrep, M, rng)
        fs = 0.0
        bs = 0.0
        cs = 0.0
        for _ in 1:nrep
            y = randn(rng)
            xs = randn(rng, M)
            fs += TSLayer.crps_ensemble(xs, y; fair = true)
            bs += TSLayer.crps_ensemble(xs, y; fair = false)
            cs += TSLayer.crps_gaussian(0.0, 1.0, y)
        end
        return (; fair = fs / nrep, biased = bs / nrep, closed = cs / nrep)
    end
    ms = mean_scores(4000, M, rng)
    @test ms.fair ≈ ms.closed rtol = 0.03
    @test !isapprox(ms.biased, ms.closed; rtol = 0.03)
    @test ms.biased > ms.fair                 # the biased form is systematically larger at small M
end

@testitem "V17 delta_rho on a known AR(1)" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test, Random
    rng = Random.MersenneTwister(7)
    # An AR(1) has rho_k = a^k exactly, so the reference is analytic.
    function ar1(a, N)
        x = zeros(N)
        s = sqrt(1 - a^2)
        for n in 2:N
            x[n] = a * x[n - 1] + s * randn(rng)
        end
        return x
    end
    N = 200_000
    a_ref, a_mod = 0.9, 0.8
    ref = reshape(ar1(a_ref, N), 1, N)
    mod = reshape(ar1(a_mod, N), 1, N)

    d1 = TSLayer.delta_rho(mod, ref; lag = 1, maxlag = 40)
    @test d1.at_lag ≈ abs(a_mod - a_ref) atol = 0.02
    d5 = TSLayer.delta_rho(mod, ref; lag = 5, maxlag = 40)
    @test d5.at_lag ≈ abs(a_mod^5 - a_ref^5) atol = 0.02

    # A model identical in law to the reference scores near zero at every lag.
    same = reshape(ar1(a_ref, N), 1, N)
    @test TSLayer.delta_rho(same, ref; lag = 1, maxlag = 40).at_lag < 0.02
    @test TSLayer.delta_rho(same, ref; lag = 5, maxlag = 40).at_lag < 0.02

    # 🔑 Why the S2' coordinate is not lag 1. At a reference rho_1 near 0.94 a model whose
    # correlation time is wrong by a factor two barely moves lag 1, while the integral-timescale
    # lag separates the two far better. Measured here rather than asserted.
    a_hi = 0.94
    a_lo = a_hi^2                     # half the correlation time
    hi = reshape(ar1(a_hi, N), 1, N)
    lo = reshape(ar1(a_lo, N), 1, N)
    tint = ceil(Int, -1 / log(a_hi))  # integral timescale in steps
    at1 = TSLayer.delta_rho(lo, hi; lag = 1, maxlag = 4tint).at_lag
    att = TSLayer.delta_rho(lo, hi; lag = tint, maxlag = 4tint).at_lag
    @test att > 2 * at1

    # #15, the whole-curve integral, needs no lag choice and is nonzero for a wrong correlation
    # time even when lag 1 is nearly right.
    @test TSLayer.delta_rho(lo, hi; lag = 1, maxlag = 4tint).integral > 5 * at1
end

@testitem "V27 a rank histogram of exchangeable draws is uniform" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test, Random, Statistics
    # 🔴 The assertion has to be about **coverage over replications**, not about one draw.
    #
    # A percentile interval is built around the observed contrast, so on calibrated data it misses
    # zero about 5% of the time by construction. An earlier version of this item asserted
    # "the interval covers zero" for a single seed and failed on a 2.5-sigma draw -- which is the
    # test being wrong, not the metric. What a calibration claim means is that the interval covers
    # zero at close to its nominal rate, and that the point estimate is unbiased with unit
    # standard deviation once the `N_eff` correction is applied.
    function null_stats(nrep, N, M, rng)
        sl = Float64[]
        cv = Float64[]
        raw = Float64[]
        cover_s = 0
        cover_c = 0
        for _ in 1:nrep
            f = exchangeable_ar1(N, M; a = 0.94, rng)
            r = TSLayer.rank_histogram(f.ens, f.truth; rng, nboot = 200)
            push!(sl, r.slope)
            push!(cv, r.convexity)
            push!(raw, r.slope_raw)
            r.slope_ci[1] <= 0 <= r.slope_ci[2] && (cover_s += 1)
            r.convexity_ci[1] <= 0 <= r.convexity_ci[2] && (cover_c += 1)
        end
        return (; sl, cv, raw, cover_s = cover_s / nrep, cover_c = cover_c / nrep)
    end

    rng = Random.MersenneTwister(20260907)
    st = null_stats(40, 4000, 10, rng)
    @info "rank histogram under the null" slope_mean=Statistics.mean(st.sl) slope_sd=Statistics.std(st.sl) raw_sd=Statistics.std(st.raw) cover_slope=st.cover_s cover_convexity=st.cover_c

    # Unbiased, and -- with the N_eff correction -- of order one rather than of order sqrt(N/N_eff).
    @test abs(Statistics.mean(st.sl)) < 0.6
    @test abs(Statistics.mean(st.cv)) < 0.6
    @test Statistics.std(st.sl) < 2.0
    # 🔴 And the uncorrected contrast is what fails: several sigma wide on calibrated data. Asserted
    # so the correction cannot be silently dropped.
    @test Statistics.std(st.raw) > 2.5
    @test Statistics.std(st.raw) > 2 * Statistics.std(st.sl)
    # Coverage near nominal. Bounds are loose because 40 replications of a 95% interval have a
    # standard error of about 3.4 percentage points.
    @test st.cover_s > 0.80
    @test st.cover_c > 0.80

    # Shape sanity on one draw: counts sum, bin count, and a genuinely correlated rank series.
    f = exchangeable_ar1(10_000, 10; a = 0.94, rng)
    rh = TSLayer.rank_histogram(f.ens, f.truth; rng, nboot = 100)
    @test rh.K == 11
    @test sum(rh.counts) == 10_000
    @test rh.n_eff < 0.6 * 10_000
    @test rh.blocklen > 1
    @test rh.eff_factor ≈ sqrt(rh.n_eff / rh.n)
    @test rh.chi2_eff ≈ rh.chi2 * rh.n_eff / rh.n
    @test rh.slope ≈ rh.slope_raw * rh.eff_factor
end

@testitem "V27 under-dispersion gives positive convexity, over-dispersion negative" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test, Random
    rng = Random.MersenneTwister(11)
    N, M = 20_000, 10
    truth = randn(rng, N)

    under = randn(rng, N, M) .* 0.5      # members too narrow: truth falls outside
    over = randn(rng, N, M) .* 2.0       # members too wide: truth sits in the middle
    ok = randn(rng, N, M)

    ru = TSLayer.rank_histogram(under, truth; rng, nboot = 300)
    ro = TSLayer.rank_histogram(over, truth; rng, nboot = 300)
    rc = TSLayer.rank_histogram(ok, truth; rng, nboot = 300)

    @test ru.convexity > 0                       # U
    @test ro.convexity < 0                       # cap
    @test ru.convexity_ci[1] > 0                 # and significantly so
    @test ro.convexity_ci[2] < 0
    @test rc.convexity_ci[1] <= 0 <= rc.convexity_ci[2]
    # The end bins are where the under-dispersed mass goes.
    @test ru.counts[1] + ru.counts[end] > 3 * ru.expected
    @test ro.counts[(M ÷ 2)] > ro.expected
    # Neither alternative is a bias, so the slope stays near zero and the two contrasts separate
    # the two failure modes rather than blurring them.
    @test abs(ru.slope) < abs(ru.convexity)
    @test abs(ro.slope) < abs(ro.convexity)

    # A shifted ensemble is the bias case: slope moves, and it moves signed.
    hi = randn(rng, N, M) .+ 1.0
    lo = randn(rng, N, M) .- 1.0
    @test TSLayer.rank_histogram(hi, truth; rng, nboot = 200).slope < 0
    @test TSLayer.rank_histogram(lo, truth; rng, nboot = 200).slope > 0
end

@testitem "V27 the raw-N chi-squared over-rejects on correlated ranks and N_eff repairs it" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test, Random, Printf
    # 🔴 The assertion is the *failure* of the uncorrected statistic, so that the correction cannot
    # later be dropped silently. Under a null of exchangeable AR(1) paths the ranks are uniform by
    # construction, so a correctly calibrated 5% test must reject about 5% of the time.
    function rejection_rates(nrep, N, M, crit, rng)
        rej_raw = 0
        rej_eff = 0
        for _ in 1:nrep
            f = exchangeable_ar1(N, M; a = 0.94, rng)
            rh = TSLayer.rank_histogram(f.ens, f.truth; rng, nboot = 0)
            rh.chi2 > crit && (rej_raw += 1)
            rh.chi2_eff > crit && (rej_eff += 1)
        end
        return (rej_raw / nrep, rej_eff / nrep)
    end
    rng = Random.MersenneTwister(4242)
    crit = 11.0705                      # chi-squared 95% quantile, dof = M = 5
    p_raw, p_eff = rejection_rates(120, 4000, 5, crit, rng)
    @info @sprintf("chi2 calibration at nominal 5%%: raw-N %.3f, N_eff %.3f", p_raw, p_eff)
    @test p_raw > 0.25                  # the uncorrected test is badly over-sized
    @test p_eff < p_raw / 2             # and the correction removes the over-sizing
    @test p_eff < 0.20
    # 🔴 Measured, and worth stating rather than hiding: the raw-N test rejects about 87% of the
    # time at a nominal 5%, and the N_eff rescaling takes that to about 0%. So the correction does
    # not *calibrate* the omnibus chi-squared, it over-shoots and makes it conservative -- a
    # chi-squared on five degrees of freedom multiplied by an N_eff/N of about 0.03 is simply
    # small. That is acceptable for a guard and useless as a discriminator, and it is the second
    # reason the Jolliffe-Primo contrasts are the primary statement: with the same correction the
    # contrasts come out unbiased with a standard deviation near 1 and coverage near nominal, which
    # the neighbouring test asserts. Do not quote `chi2_eff` as a calibrated p-value.
end

@testitem "V26 level and increment targets give identical ranks" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test, Random
    # `metrics.md` §9 specification 1: for the one-step histogram the level and increment ranks are
    # provably identical, because given the history `dQ = q - q*` is a deterministic shift and a
    # shift common to the truth and every member cannot move a rank. Asserted once so that
    # reporting `dQ` throughout needs no further argument.
    #
    # ⚠️ It does **not** extend to a free-running run, where `q*` is produced by the solver and
    # responds to what the model did. There the level is pinned by the replayed predictor and `dQ`
    # is the only discriminating quantity for a different reason (O7).
    rng = Random.MersenneTwister(3)
    N, M = 500, 20
    q_star = randn(rng, N) .* 3 .+ 10
    truth_q = q_star .+ randn(rng, N)
    ens_q = q_star .+ randn(rng, N, M)

    r_level = TSLayer.ranks(ens_q, truth_q; rng = Random.MersenneTwister(9))
    r_incr = TSLayer.ranks(ens_q .- q_star, truth_q .- q_star; rng = Random.MersenneTwister(9))
    @test r_level == r_incr

    # The histogram built from either is therefore the same object.
    a = TSLayer.rank_histogram(ens_q, truth_q; rng = Random.MersenneTwister(9), nboot = 0)
    b = TSLayer.rank_histogram(ens_q .- q_star, truth_q .- q_star;
                               rng = Random.MersenneTwister(9), nboot = 0)
    @test a.counts == b.counts
    @test a.convexity == b.convexity
end

@testitem "ties are broken at random, because the clamp makes exact duplicates" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test, Random
    # When the stabiliser fires every member returns the identical dQ = 0 and the truth ties with
    # all of them. Breaking ties to one end would manufacture the U this metric looks for.
    N, M = 4000, 5
    ens = zeros(N, M)
    truth = zeros(N)
    rlow = TSLayer.ranks(ens, truth; ties = :low)
    rhigh = TSLayer.ranks(ens, truth; ties = :high)
    rrand = TSLayer.ranks(ens, truth; ties = :random, rng = Random.MersenneTwister(5))
    @test all(==(1), rlow)                       # every tie to the bottom bin
    @test all(==(M + 1), rhigh)                  # every tie to the top bin
    @test length(unique(rrand)) == M + 1         # spread over all bins
    jp = TSLayer.jolliffe_primo([count(==(k), rrand) for k in 1:(M + 1)])
    @test abs(jp.convexity) < 3                  # and no manufactured U
    @test TSLayer.jolliffe_primo([count(==(k), rlow) for k in 1:(M + 1)]).convexity > 10
end

@testitem "spread-skill needs its finite-M correction" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test, Random
    rng = Random.MersenneTwister(20260907)
    K, nq = 40_000, 1
    for M in (5, 10)
        # A perfect ensemble: truth and members drawn from the same law about a common mean.
        ens = randn(rng, K, nq, M)
        truth = randn(rng, K, nq)
        corrected = TSLayer.spread_skill(ens, truth; correct = true)
        raw = TSLayer.spread_skill(ens, truth; correct = false)
        @test corrected.ratio ≈ 1 atol = 0.02
        @test raw.ratio ≈ sqrt(M / (M + 1)) atol = 0.02
        @test corrected.correction ≈ sqrt((M + 1) / M)
    end
    # 🔴 The size of the problem at the archive's M = 5: an uncorrected perfect ensemble reads
    # 0.913, not the 0.953 both documents quote for M = 10. Against a [0.8, 1.25] band that eats a
    # third of the lower margin.
    @test sqrt(5 / 6) ≈ 0.9128709291752769 rtol = 1e-12
    @test sqrt(10 / 11) ≈ 0.9534625892455922 rtol = 1e-12

    # Direction: an over-confident ensemble reads below 1, an over-dispersed one above.
    ens = randn(rng, K, nq, 10) .* 0.5
    truth = randn(rng, K, nq)
    @test TSLayer.spread_skill(ens, truth).ratio < 1
    ens2 = randn(rng, K, nq, 10) .* 2
    @test TSLayer.spread_skill(ens2, truth).ratio > 1
end

@testitem "summed KS and ensemble KS are different objects" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test, Random
    rng = Random.MersenneTwister(17)
    nq, T = 2, 8000
    ref = randn(rng, nq, T)
    # Replicas individually biased in opposite directions: collectively they cover the reference,
    # individually none does. Summed KS sees the defect; ensemble KS does not.
    trajs = [randn(rng, nq, T) .+ s for s in (-1.0, 1.0)]
    per_replica = [TSLayer.summed_ks(t, ref).total for t in trajs]
    ens = TSLayer.ensemble_ks(trajs, ref).total
    @test all(>(0.3), per_replica)
    @test ens < minimum(per_replica) / 2
    # Which is why they are reported side by side and never averaged together.
    @test ens != sum(per_replica) / length(per_replica)

    # KS is blind to order: shuffling a series leaves it unchanged. This is metric #10's whole
    # limitation and the reason it is a guard rather than a selector.
    shuffled = ref[:, randperm(rng, T)]
    @test TSLayer.summed_ks(shuffled, ref).total == 0.0
    @test TSLayer.delta_rho(shuffled, ref; lag = 1, maxlag = 20).at_lag >= 0.0

    # The reference-vs-reference noise floor is a real, nonzero scale: differences below it are not
    # model differences.
    floor_ = TSLayer.ks_noise_floor(ref)
    @test 0 < floor_.total < minimum(per_replica)
end

@testitem "clamp census counts the record, not the model" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test
    q_star = [1.0 0.001 1.0 1.0; 1.0 1.0 1.0 0.0005]
    c = TSLayer.clamp_census(q_star)
    @test c.nsteps == 4
    @test c.nfired == 2                       # steps 2 and 4, one QoI each
    @test c.rate == 0.5
    @test c.per_qoi_rate == [0.25, 0.25]
    @test c.fired == [false, true, false, true]
    # The training threshold on Taylor-Green is half the deployment one, so a band of steps is
    # trained on and then clamped.
    @test TSLayer.clamp_census(q_star; threshold = 0.5e-2).nfired == 2
    @test TSLayer.clamp_census([0.007 1.0], threshold = 1e-2).nfired == 1
    @test TSLayer.clamp_census([0.007 1.0], threshold = 0.5e-2).nfired == 0
end

@testitem "stability fraction finds the first non-finite column" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test
    good = ones(2, 10)
    bad = ones(2, 10); bad[1, 6] = NaN
    inf_ = ones(2, 10); inf_[2, 3] = Inf
    s = TSLayer.stability_fraction([good, bad, inf_])
    @test s.n == 3
    @test s.nstable == 1
    @test s.fraction ≈ 1 / 3
    @test s.first_nan == [nothing, 6, 3]
    # A shorter horizon can make a late blow-up invisible, which is why the horizon is an argument.
    @test TSLayer.stability_fraction([good, bad, inf_]; horizon = 5).nstable == 2
end

@testitem "autocorr treats a constant series as constant" default_imports = false setup = [TSLayer, ScoreFixture] begin
    using Test, Random
    # 🔴 Regression test for a real defect in the pre-existing `autocorr`. It guarded degeneracy
    # with `den == 0`, which almost never fires: on `fill(3.7, 1000)` the centred sum of squares is
    # 2.8e-26 rather than exactly zero, because `sum(x)/n` does not reproduce `x` bit-for-bit. The
    # autocorrelation was then computed from rounding noise and returned `[1.0, 0.999]` --
    # indistinguishable from a strongly autocorrelated series.
    #
    # It mattered at once: the data-driven noise model's predictive mean IS a constant, so the
    # scoring driver reported its lag-1 autocorrelation as 1.0000 when the truthful answer is that
    # it has none. A metric whose whole job is to expose "no dynamics" was reporting maximal
    # dynamics for the one model that certainly has none.
    for c in (3.7, 1e-4, 1e6, 0.0, -2.5)
        @test all(iszero, TSLayer.autocorr(fill(c, 1000), 3))
    end

    # And the guard must be relative, not magnitude-scaled: an AR(1) sitting on a huge offset is
    # not constant, and a tiny-amplitude AR(1) is not constant either. Scaling the threshold by
    # `maximum(abs, x)` gets both of these wrong.
    rng = Random.MersenneTwister(31)
    a = 0.9
    y = zeros(20_000)
    for n in 2:20_000
        y[n] = a * y[n - 1] + sqrt(1 - a^2) * randn(rng)
    end
    base = TSLayer.autocorr(y, 2)
    @test base[1] ≈ 1
    @test base[2] ≈ a atol = 0.02
    @test base[3] ≈ a^2 atol = 0.03
    # The contract is stated, not derived from a representability argument: a series whose centred
    # RMS is below `CONSTANT_RTOL` times its largest absolute value counts as constant. That is a
    # deliberately conservative policy -- at a relative variation of 1e-15 the signal is only about
    # nine times the centring's own rounding error, so its autocorrelation would be mostly noise --
    # and asserting the policy is what keeps the threshold from drifting.
    for scale in (1.0, 1e-9, 1e9), offset in (0.0, 1e8, -1e6)
        z = y .* scale .+ offset
        got = TSLayer.autocorr(z, 2)
        rel = scale / max(abs(offset), scale)                 # the series' own relative variation
        if rel > 1e3 * TSLayer.CONSTANT_RTOL
            @test got ≈ base rtol = 1e-8                      # affine-invariant well inside
        elseif rel < TSLayer.CONSTANT_RTOL
            @test all(iszero, got)                            # and constant well outside
        end
    end
    # Scale alone never triggers it -- only variation relative to magnitude does.
    @test TSLayer.autocorr(y .* 1e-9, 2) ≈ base rtol = 1e-8
    @test TSLayer.autocorr(y .* 1e9, 2) ≈ base rtol = 1e-8
    # An offset large enough to drown the signal does, and 1e-9 on 1e8 is far past it: eps(1e8) is
    # about 1.5e-8, so every sample rounds into a handful of values.
    z = y .* 1e-9 .+ 1e8
    @test all(iszero, TSLayer.autocorr(z, 2))
    @test length(unique(z)) < length(z) ÷ 2
    @test TSLayer.CONSTANT_RTOL == 1e-12
end
