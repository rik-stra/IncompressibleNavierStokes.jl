# Test harness for the RikFlow time-series layer (V0).
#
# Run it as
#
#     julia --project=lib/RikFlow/test lib/RikFlow/test/runtests.jl
#
# and NOT via `Pkg.test`. The reason is in `test/Project.toml`: this suite deliberately does not
# depend on RikFlow, because the `ts_*.jl` files are stdlib-only and the tests include them
# directly. `Pkg.test` would force the package under test to load, dragging in
# IncompressibleNavierStokes, CUDA, Makie and Lux for tests that need none of them.
#
# Why this file exists at all: the 2026-09-05 session lost two normal-equation bugs and four claims
# to the absence of a test directory, one of them a relative error of 5.8 in the fitted
# coefficients against paper 2's archive. The cost of not having this is measured, not
# hypothetical (`meta_files/plan.md` §25).

using TestItemRunner

# Only run tests from this test dir, and not from other packages in the monorepo.
@run_package_tests filter = t -> occursin(@__DIR__, t.filename)
