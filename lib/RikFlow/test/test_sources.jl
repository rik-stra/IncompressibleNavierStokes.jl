# V31 -- source-level checks on the scripts the offline suite cannot load.
#
# `exp_square_HIT/tools/run_d6.jl` needs CUDA and IncompressibleNavierStokes, which this suite
# deliberately does not depend on (see `runtests.jl`), so nothing here ever macroexpanded it. That
# gap cost a Snellius run: `@printf` requires its format argument to be a *single string literal*,
# a `"..." * "..."` concatenation parses fine and fails only at macroexpansion, and for a
# documented function macroexpansion happens when the **docstring** is processed. So the error
# pointed at the docstring of `run_ic` (line 231) while the defect was 140 lines further down, and
# it appeared only on the cluster, after the copy and the queue.
#
# The check is therefore on the parsed source, not on a loaded module: it needs no packages and so
# covers every script in the tree, including the drivers this suite cannot import.
@testitem "V31 every @printf format argument is a string literal" default_imports = false begin
    using Test

    root = normpath(joinpath(@__DIR__, "..", ".."))   # code_base/lib

    """Walk an AST and collect `@printf`/`@sprintf` calls whose format argument is not a literal."""
    function bad_formats(ex, file, acc = String[])
        ex isa Expr || return acc
        if ex.head === :macrocall && ex.args[1] isa Symbol &&
           String(ex.args[1]) in ("@printf", "@sprintf")
            rest = ex.args[3:end]
            if !isempty(rest)
                # the io argument is optional, so the format is the first or the second argument
                fmt = rest[1] isa String ? rest[1] : (length(rest) > 1 ? rest[2] : nothing)
                if !(fmt isa String)
                    ln = ex.args[2]
                    push!(acc, string(relpath(file, root), ":",
                                      ln isa LineNumberNode ? ln.line : "?", "  ",
                                      String(ex.args[1]), " fmt = ", sprint(show, fmt)))
                end
            end
        end
        for a in ex.args
            bad_formats(a, file, acc)
        end
        return acc
    end

    """Parse every `.jl` file under `root` and return the offenders plus the file count."""
    function scan(root)
        bad = String[]
        nfiles = 0
        for (dir, _, files) in walkdir(root), f in files
            endswith(f, ".jl") || continue
            p = joinpath(dir, f)
            ast = try
                Meta.parseall(read(p, String); filename = p)
            catch
                continue      # a file this Julia cannot parse is not this test's business
            end
            nfiles += 1
            bad_formats(ast, p, bad)
        end
        return bad, nfiles
    end

    bad, nfiles = scan(root)

    @test nfiles > 50                     # the walk found the tree, not an empty directory
    @test isempty(bad) || (println("\n  ", join(bad, "\n  ")); false)

    # Positive control: a scanner that has never seen an offender is not a test. The first three
    # are the exact shape that broke run_d6.jl on Snellius; the last three are what is fine.
    parse1(code) = bad_formats(Meta.parseall(code; filename = "control.jl"), "control.jl")
    @test length(parse1(raw"""@printf("a %d b " * "c %d\n", 1, 2)""")) == 1
    @test length(parse1(raw"""@printf(io, "a %d " * "b\n", 1)""")) == 1
    @test length(parse1(raw"""@sprintf("%d" * "%d", 1, 2)""")) == 1
    @test isempty(parse1(raw"""@printf("a %d b c %d\n", 1, 2)"""))
    @test isempty(parse1(raw"""@printf(io, "a %d\n", 1)"""))
    @test isempty(parse1(raw"""@printf(stdout, "plain\n")"""))
end
