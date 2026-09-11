# D6 on Snellius — what to copy, what to run

The operational half of **P2c / D6**: the multi-IC ensemble that metric #17 (lead-resolved
spread–skill) and RH-3 (lead-resolved rank histograms) need. Design and rationale live in
`meta_files/handoff_p2c_d6.md`; the numbers, once there are any, go in
`lib/RikFlow/analysis/results.md`.

**State at time of writing (2026-09-11):** everything is built, the offline suite is green at
**1482 tests**, and **steps 1 and 2 have both run on GPU on Snellius.** The smoke passed; the
validation run (ordinal 0, 5 members, 1308 steps) passed its gate **5/5** with `dQ` bit-identical
to paper 2's archive across all 100 replayed columns. Two things came out of it:

- ✅ **The GPU random stream is unchanged** under CUDA 5.6.1 → 5.11.3 — `claude_memory.md` gotcha
  **#43**, which only this run could answer, is closed.
- 🔴 **The validation's acceptance criterion had to be rewritten**, because the original one was
  unreachable by construction and failed a correct run. Gotcha **#48**, and step 2 below.

🔴 **What is left: `--array=1-5`, then `--array=1-180`. No scored run exists yet, so there is still
no D6 number.**

✅ **Julia 1.13 is no longer a risk.** Snellius' juliaup default is 1.13.0 and the *old* pinned
stack failed there twice (CUDA's `libnvml.jl` `ccall` parsing; `MakieCore` on `Core.TypeName.mt`);
the dependency surgery of 2026-09-10 removed both causes, and the stack has now run on 1.13 on the
GPU nodes. Fallback if it ever regresses: the archive's own environment is on disk at
`julia_code/IncompressibleNavierStokes.jl/lib/RikFlow/Manifest.toml` (Julia 1.11.2, CUDA 5.6.1) —
copy it in and use `julia +1.11`, adding `+1.11` to `batch_scripts/run_d6.sh` too.

⚠️ `Distributions` and `PDMats` are **pinned** (`=0.25.117`, `=0.11.32`). Do not lift them: the
archived `LinReg.jld2` stores its residual model as a two-parameter `PDMat`, and a newer PDMats
makes `rand(rng, stoch_distr)` — which every deployed step calls — fail. See `claude_memory.md`
gotcha #42.

---

## What to copy — 19.9 MB, eight files

| file | size | why |
|---|---|---|
| `d6_ic_validation.jld2` | 3.31 MB | ordinal 0 — the archived runs' own initial condition |
| `d6_ic_manifest.jld2` | 21 kB | 🔑 **not optional.** `ic_dir()` locates the directory *by* this file, and `load_ic` cross-checks it against `select_ics` — that check is what catches an IC set built with a different `K`, `nlead` or record from the one the driver assumes |
| `d6_ic_42.jld2` … `d6_ic_50.jld2` | 3.31 MB each | the pilot's five, ordinals 1–5 |
| `TO_LRS/LinReg1/LinReg.jld2` | 13 kB | the fitted M0 |

The first seven are in `lib/RikFlow/analysis/output/d6_ics/` (build them with
`julia --startup-file=no --project=analysis analysis/build_d6_ics.jl`, which writes all 180 plus the
validation package).

**Deliberately not copied.**

- `TO_LRS/LinReg1/parameters.jld2` — unnecessary. `LinReg.jld2` itself carries `hist_len = 5` and
  `hist_var = :q_star_q`, and `run_d6.jl` reads them from there.
- the 1.29 GB tracked record and the 1.39 GB HF reference — **scoring happens locally** after the
  runs come back, so the truth never needs to be on Snellius. `build_d6_ics.jl` pre-slices the
  record once, on the workstation, which is the whole reason for one file per IC.

```bash
# from lib/RikFlow on the workstation; $SNEL = login, $REPO = fork root there
D=$REPO/lib/RikFlow/exp_square_HIT/output
ssh $SNEL "mkdir -p $D/d6_ics $D/TO_LRS/LinReg1 $D/D6"
scp analysis/output/d6_ics/d6_ic_{validation,manifest,42,44,46,48,50}.jld2  $SNEL:$D/d6_ics/
scp "/c/Users/rik/Documents/julia_code/INS_paper_summer2025/lib/RikFlow/exp_square_HIT/output/paper_data_HIT/TO_LRS/LinReg1/LinReg.jld2" \
    $SNEL:$D/TO_LRS/LinReg1/
```

Those are `run_d6.jl`'s default locations, so no environment variables are needed. To put them
elsewhere: `D6_IC_DIR`, `D6_MODEL`, `D6_OUT`, `D6_MEMBERS`.

Code comes over with `git pull` — the analysis scripts, `tools/` and `batch_scripts/run_d6.sh` are
all committed (`e41cadd` and later).

---

## What to run

```bash
cd $REPO/lib/RikFlow/exp_square_HIT
export JULIA_DEPOT_PATH=$HOME/julia/julia_a1003:      # trailing colon, as every script here has it
julia --project -e 'using Pkg; Pkg.instantiate()'
```

⚠️ The depot matches the partition: `julia_a1003` for `gpu_a100`, `julia_h100` for `gpu_h100`.
`batch_scripts/run_d6.sh` sets the a100 one.

### 1 · Smoke test — first, and it is cheap  ✅ passed on GPU 2026-09-11

```bash
julia --project tools/smoke_d6.jl
```

400 steps, one member, ~0.01 SBU. Asserts completion, every expected output key, `q` with
`nt + 1` columns, no NaN, the clamp count, and two things that matter more than the rest.

**What it gave:** `q0` to 6.0e-04, no `fields` in a 36 kB output, clamp **0** of 300,
`ou_advance` moving the trajectory by **0.503** — the same value the CPU run gave.

⚠️ **Do not take a cost figure from this step.** At `M = 1` the whole run is compilation: it
printed 62.55 s/TU against a steady stepping rate of 3.56 s/TU, a factor 18 (`claude_memory.md`
gotcha #44). `run_d6.jl` now refuses to quote a rate at `M = 1` for that reason. And note that even
3.56 is not the planning number — it is stepping only, where **4.102 s/TU** (step 2, `M = 5`) is
wall per member including setup and the write. Budget with 4.102.

The asserts:

- 🔴 **that no velocity fields were written to the output file.** `params_track` carries
  `savefreq = 100`; at 1308 steps that is ~13 snapshots × 3.3 MB per run, i.e. **80 GB** over 1800
  runs against **160 MB** of QoIs.
  ⚠️ Note precisely what holds: `savefreq = nt + 1` leaves **one** `t = 0` field in memory, not
  zero. `qoisaver`'s initializer does `state[] = state[]` (`RikFlow.jl:332`) — which is what gives
  `q` its `nstep+1` columns, and therefore the offset the whole index alignment rests on — and
  `fieldsaver` is registered before it, so it fires at `n = 0` where `0 % anything == 0`. What
  keeps the disk cost at zero is that `run_d6.jl`'s `jldsave` writes no `fields` key at all.
  `run_ic` bounds the in-memory count at that one snapshot; the smoke test asserts the file has
  none.
- 🔴 **that `ou_advance` actually changes the trajectory.** The unit tests prove the replay is
  correct arithmetic and `analysis/ou_replay.jl` proves the advance count is right, but neither goes
  through `online_sgs`. If the keyword were dropped between the driver and `Setup`, every member's
  forcing would be out of phase with its own initial condition, the spread–skill ratio would be
  biased *downward* — toward a false "over-confident" verdict — and nothing would say so.

### 2 · Validation — one task, before the pilot  ✅ passed on GPU 2026-09-11

```bash
D6_MEMBERS=5 julia --project tools/run_d6.jl 0        # or sbatch with --array=0
```

Ordinal 0 is `fields[1]` of the **10 TU** tracked record — the initial condition every archived
online run launched from (`paper_runs/online_sgs.jl:50`; line 52 has the 100 TU file commented out).
So `n_k = 0`, `ou_advance = 0` (the identity point of the replay), and the model seeds are the
archive's own `Xoshiro(236 + member)`. `D6_MEMBERS=5` matches the archive's five replicas; the
default 10 works too and `compare_validation` reports the extra members as having no counterpart.

Verified before any GPU time: the validation package's `q_at_ic` is bit-identical to **all five**
archived replicas' `q[:, 1]`, and its `dQ_warm` to the archive's literal `dQ[:, 1:100]`.

🔴 **What this step can and cannot ask for — read this before reading its output.** It is tempting
to state the check as *"its `q` must reproduce the archived replica's first 1309 columns"*. That was
the original wording, it was the original acceptance criterion, and it is **wrong in two independent
ways** (`claude_memory.md` gotcha **#48**):

1. **The statistic saturates.** Two *archived* replicas of LinReg1 — same code, same inputs, only
   the seed differs — are **1.00 apart** in per-QoI relative rms over those 1309 columns (range
   0.65–1.55 over the 10 pairs), because two draws from one stationary law are √2 apart by
   construction. A `1e-2` threshold on it is unreachable however correct the code is.
2. **The run is not the same dynamical system as the archive.** Commit `09954be1` zeroes the
   Nyquist wavenumber before `∂` is built, and `∂` feeds `get_vi_functions`, so the Z-QoI direction
   vectors and hence `tau` changed with it (gotchas #45, #46). Every archived record predates it.

So the criterion is now **two windows, and they test different things**:

- 🔒 **GATE — the replayed warm-up, columns `1:nwarm`.** There the sampler emits `spinnup_data`
  verbatim, so `dQ[:, 1:100]` must be **bit-identical** to the archive's own slice. That is exact,
  and it is the sharpest single check in the D6 path: it proves the warm-up slice, the history
  layout and the deployment wiring at once. With `dQ` pinned, `q`'s drift over the same window is
  the solver, the OU forcing and `tau` alone — gated at `1e-2` of a sd.
- **REPORTED, never gated** — the divergence column, and the full-window rms printed *beside*
  `replica_spread`'s archive-to-archive yardstick so a saturated number is never shown bare.

⚠️ **The window in which this step tests the *sampler* is only ~50 columns wide**: `1:100` are
replayed and test nothing about it, and by ~150 the pair has decorrelated. That is a property of the
design, not a defect, and it is why the pilot is scored against the HF reference instead.

**What it gave, 2026-09-11:** gate **5/5**, `dQ` bit-identical on all 100 columns for all five
members, max `q` deviation 3.9e-3 against the 1e-2 gate, divergence crossing 0.1 at column
**104–105** — four to five steps after the first *sampled* `dQ`.

✅ **And it closed the one question only it could answer: the GPU random stream.** The CPU streams
were already known unchanged — 25 chained `randn!(Xoshiro(333), ::Matrix{Float32})` draws are
byte-identical on Julia 1.11.9, 1.12.7 and 1.13.0 — and `Distributions`/`PDMats` are pinned so the
`rand(rng, ::MvNormal)` path is unchanged too. But in production `z` is a `CuArray`, so `randn!`
runs through CUDA.jl, which went **5.6.1 → 5.11.3**, and a shifted device stream would have moved
the OU forcing while every CPU check still passed. It did not: at **column 2 — one step** — the five
non-`Z[16,32]` QoIs agree with the archive to **≤ 5.6e-07 of a sd**, and the deviation then grows
smoothly 5.6e-7 → 2.3e-5 → 3.4e-4 → 3.9e-3. Round-off amplification, not a different draw. Gotchas
#42 and #43, the latter now closed.

### 3 · Pilot

```bash
sbatch batch_scripts/run_d6.sh                        # --array=1-5 as committed
```

`$SLURM_ARRAY_TASK_ID` is the **ordinal**, not the field index `k`; `run_d6.jl` maps it through
`select_ics`, so ordinals 1–5 are `k = 42, 44, 46, 48, 50`.

### 4 · Pull back, and check the validation *before* reading anything else

```bash
scp $SNEL:$D/D6/'*.jld2' exp_square_HIT/output/D6/    # from lib/RikFlow on the workstation
julia --startup-file=no --project=analysis analysis/score_d6.jl
```

`compare_validation` runs first, deliberately: if the D6 path does not reproduce the archive over
the window that **has** a right answer, nothing scored below it is worth reading. Read its **GATE**
block as the verdict and its **REPORTED** block as description — step 2 says why, and the gate is
already known to pass 5/5.

⚠️ **A full-window rms near 1.0 is saturation, not a defect.** `replica_spread` prints the
archive's own replica-to-replica value beside it for exactly that reason. Do not reintroduce a
threshold on that statistic; **V32** in `test/test_d6_score.jl` exists to stop it coming back.

📈 Figures for this comparison:
`julia --startup-file=no --project=analysis analysis/plot_validation.jl` writes
`analysis/figures/d6_validation_{trajectories,divergence}.png`. The divergence panel is the one the
verdict is read from.

✅ **The cost figure is already measured** — `meta_files/handoff_p2c_d6.md` §2 carries it:
**4.102 s/TU** steady (13.4 s per member, 1308 steps), ≈59 s compilation once per array task,
**≈193 s per task**, **≈9.65 GPU-hours** for all 180, of which **31% is compilation**. 99 kB per
member. So from the pilot just **confirm** those, and 🔴 **look up this partition's
SBU-per-GPU-hour rate** (`accinfo` / `budget-overview`): plan's two SBU figures differ by 10×
(§13 P2 says 1 TU ≈ 0.01 SBU; P2c's cost model uses 0.1023 SBU/TU) and only that rate settles it.
Confirm too that the files carry no `fields` key.

K = 5 proves the pipeline and the index alignment, not a histogram — five instances against eleven
bins.

### 5 · Full run

```bash
scp analysis/output/d6_ics/d6_ic_*.jld2 $SNEL:$D/d6_ics/    # the remaining 175; 596 MB for all 180
# in batch_scripts/run_d6.sh: --array=1-5  ->  --array=1-180   (add %20 to cap concurrency)
sbatch batch_scripts/run_d6.sh
```

**Nothing else changes.** The ordinal → `k` map is `select_ics`'s and is deterministic, so the
pilot's five stay the same five and no result is renumbered. **≈9.65 GPU-hours and ≈178 MB back**,
both measured rather than estimated (step 4).

---

## Three traps worth repeating

⚠️ **`d6_valid_ic1_*` is `k = 1`, not ordinal 1.** The validation package hardcodes `k = 1`
(`build_validation_ic`), and output filenames carry `k`, not the ordinal. Scored ICs start at
`k = 42`, so there is no collision — but when you `scp` the validation results back, the glob is
`d6_valid_ic1_m*.jld2` and it has nothing to do with ordinal 1.

⚠️ **An empty `.out` means not started, not hung.** Julia buffers stdout when redirected; every
progress line in `run_d6.jl` and `smoke_d6.jl` is followed by `flush(stdout)`. Two healthy jobs have
already been killed on that misreading.

⚠️ **The validation IC is not part of the scored set, and must not become part of it.** `t = 0` is
inside M0's fit window, so its short-lead spread would be measured on data the conditional mean has
already seen; and V28 requires D6's set to be disjoint from the archived runs' IC. The separation is
by filename — `d6_valid_ic1_m*.jld2` against `d6_online_ic<k>_m*.jld2` — so the scorer's glob cannot
see it and no filtering step can be forgotten.

## Without any data

```bash
julia --startup-file=no --project=analysis analysis/score_d6.jl --preview
```

Prints the per-QoI lead grids and the index alignment worked out for one IC, so the design can be
read rather than trusted. The grids, for reference — `{0.25, 0.5, 1, 2, 5, 10} × T_int(i)`, in
steps:

| QoI | `T_int` [TU] | leads [steps] |
|---|---|---|
| Z[0,6] | 0.1118 | 11, 22, 45, 89, 224, 447 |
| E[0,6] | 0.0082 | 1, 2, 3, 7, 16, 33 |
| Z[7,15] | 0.0923 | 9, 18, 37, 74, 185, 369 |
| E[7,15] | 0.0669 | 7, 13, 27, 54, 134, 268 |
| Z[16,32] | 0.2926 | 29, 59, 117, 234, 585, 1170 |
| E[16,32] | 0.3017 | 30, 60, 121, 241, 603, **1207** |

35 distinct leads in the union; the longest is 1207 of the 1208 available, which is what set the
forecast length — one more multiple of `T_int` would not fit.
