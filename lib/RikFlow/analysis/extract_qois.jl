# Extract the QoI time series from a tracking run into a small, self-contained file.
#
# A tracking file is dominated by something none of the time-series work reads. `track_ref` returns
# `(; dQ, tau, q, q_star, fields)` (`INS/lib/RikFlow/src/LFsims.jl:75`) and `fields` is the velocity
# history written by `fieldsaver(; nupdate = savefreq)`. On the 100 TU HIT record that is 400
# snapshots of a 64^3 x 3 field, about 1.26 GB of a 1.3 GB file. The QoI arrays are ~1 MB each.
#
# JLD2 stores `data_track` as one compound dataset, not a group, so there is no way to read `q`
# without materialising `fields` alongside it -- which is slow enough to dominate a run and large
# enough to exhaust memory when two jobs overlap. The fix is to pay that read once.
#
# Usage:
#   julia --project=<env> analysis/extract_qois.jl <tracking.jld2> [more.jld2 ...]
#
# Writes `analysis/data/<basename>_qois.jld2` holding only the QoI arrays plus provenance.

using JLD2, Printf, Dates

const DATA_DIR = joinpath(@__DIR__, "data")

"""
    qoi_path(file, dir = DATA_DIR)

Where the extracted QoIs for `file` live. Keyed on the source basename so two records never
collide, and kept inside this project rather than beside the source, which belongs to another repo.
"""
qoi_path(file, dir = DATA_DIR) =
    joinpath(dir, replace(basename(file), r"\.jld2$" => "") * "_qois.jld2")

"""
    extract_qois(file; dir = DATA_DIR, force = false)

Read a tracking file once and write its QoI arrays to a small file. Returns the output path.
Existing output is kept unless `force`, so this is safe to call from a driver.
"""
function extract_qois(file; dir = DATA_DIR, force = false)
    out = qoi_path(file, dir)
    if isfile(out) && !force
        @printf("  exists, skipping: %s (%.2f MB)\n", basename(out), filesize(out) / 2^20)
        return out
    end
    isfile(file) || error("no such tracking file: $file")
    mkpath(dir)

    @printf("  reading %s (%.2f GB) ...\n", basename(file), filesize(file) / 2^30)
    t0 = time()
    payload = jldopen(file, "r") do io
        key = "data_track" in keys(io) ? "data_track" :
              "data_train" in keys(io) ? "data_train" :
              error("no data_track or data_train in $file; keys = $(keys(io))")
        d = io[key]
        # Take whichever of the QoI arrays this record carries; `tau` and `dQ` are absent from some
        # of the older tracking scripts, and nothing here should fail because of that.
        got = Dict{String,Any}("key" => key)
        for f in (:q, :q_star, :dQ, :tau)
            hasproperty(d, f) && (got[string(f)] = Array(getproperty(d, f)))
        end
        haskey(got, "q") && haskey(got, "q_star") ||
            error("$key lacks q and/or q_star; fields = $(propertynames(d))")
        got
    end
    dt = time() - t0

    payload["source"] = abspath(file)
    payload["source_bytes"] = filesize(file)
    payload["extracted"] = string(now())
    jldsave(out; (Symbol(k) => v for (k, v) in payload)...)

    @printf("  read in %.1f s; wrote %s (%.2f MB, %.0fx smaller)\n", dt, basename(out),
            filesize(out) / 2^20, filesize(file) / max(filesize(out), 1))
    for f in ("q", "q_star", "dQ", "tau")
        haskey(payload, f) && @printf("    %-8s %s\n", f, size(payload[f]))
    end
    return out
end

"""
    load_qois(file)

Load the QoI arrays for a tracking run, extracting them first if that has not been done. Drivers
should call this rather than opening a tracking file directly.
"""
function load_qois(file)
    p = endswith(file, "_qois.jld2") ? file : extract_qois(file)
    d = load(p)
    return (; q = d["q"], q_star = d["q_star"], key = d["key"],
            dQ = get(d, "dQ", nothing), tau = get(d, "tau", nothing),
            source = get(d, "source", p))
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: extract_qois.jl <tracking.jld2> [more.jld2 ...]")
    println("extracting QoIs to ", DATA_DIR)
    for f in ARGS
        println()
        extract_qois(f)
    end
end
