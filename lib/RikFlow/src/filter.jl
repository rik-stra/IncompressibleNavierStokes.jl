# Functions to coarse-grain high-fidelity velocity field to coarse velocity field.

"""
Discrete DNS filter.

Subtypes `ConcreteFilter` should implement the in-place method:

    (::ConcreteFilter)(v, u, setup_les, compression)

which filters the DNS field `u` and put result in LES field `v`.
Then the out-of place method:

    (::ConcreteFilter)(u, setup_les, compression)

automatically becomes available.
"""
abstract type AbstractFilter end

"Average fine grid velocity field over coarse volume face."
struct FaceAverage <: AbstractFilter end

"Average fine grid velocity field over coarse volume."
struct VolumeAverage <: AbstractFilter end

(Φ::AbstractFilter)(u, setup_les, compression) =
    Φ(vectorfield(setup_les), u, setup_les, compression)

function (::FaceAverage)(v, u, setup_les, comp)
    (; grid, backend, workgroupsize) = setup_les
    (; dimension, Nu, Iu) = grid
    D = dimension()
    @kernel function Φ!(v, u, ::Val{α}, face, I0) where {α}
        I = @index(Global, Cartesian)
        J = I0 + comp * (I - oneunit(I))
        s = zero(eltype(v))
        for i in face
            s += u[J+i, α]
        end
        v[I0+I, α] = s / comp^(D - 1)
    end
    for α = 1:D
        ndrange = Nu[α]
        I0 = getoffset(Iu[α])
        face = CartesianIndices(ntuple(β -> β == α ? (comp:comp) : (1:comp), D))
        Φ!(backend, workgroupsize)(v, u, Val(α), face, I0; ndrange)
    end
    v
end

"Reconstruct DNS velocity `u` from LES velocity `v`."
function reconstruct!(u, v, setup_dns, setup_les, comp)
    (; grid, boundary_conditions, backend, workgroupsize) = setup_les
    (; dimension, N) = grid
    D = dimension()
    e = Offset(D)
    @assert all(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions)
    @kernel function R!(u, v, ::Val{α}, volume) where {α}
        J = @index(Global, Cartesian)
        I = oneunit(J) + comp * J
        J = oneunit(J) + J
        Jleft = J - e(α)
        Jleft.I[α] == 1 && (Jleft += (N[α] - 2) * e(α))
        for i in volume
            s = zero(eltype(v[α]))
            s += (comp - i.I[α]) * v[J, α]
            s += i.I[α] * v[Jleft, α]
            u[I-i, α] = s / comp
        end
    end
    for α = 1:D
        ndrange = N .- 2
        volume = CartesianIndices(ntuple(β -> 0:comp-1, D))
        R!(backend, workgroupsize)(u, v, Val(α), volume; ndrange)
    end
    u
end

"Reconstruct DNS velocity field. See also [`reconstruct!`](@ref)."
reconstruct(v, setup_dns, setup_les, comp) =
    reconstruct!(vectorfield(setup_dns), v, setup_dns, setup_les, comp)

function (::VolumeAverage)(v, u, setup_les, comp)
    (; grid, boundary_conditions, backend, workgroupsize) = setup_les
    (; dimension, N, Nu, Iu) = grid
    D = dimension()
    @assert all(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions)
    @kernel function Φ!(v, u, ::Val{α}, volume, I0) where {α}
        I = @index(Global, Cartesian)
        J = I0 + comp * (I - oneunit(I))
        s = zero(eltype(v))
        # n = 0
        for i in volume
            # Periodic extension
            K = J + i
            K = mod1.(K.I, comp .* (N .- 2))
            K = CartesianIndex(K)
            s += u[K, α]
            # n += 1
        end
        n = (iseven(comp) ? comp + 1 : comp) * comp^(D - 1)
        v[I0+I, α] = s / n
    end
    for α = 1:D
        ndrange = Nu[α]
        I0 = getoffset(Iu[α])
        volume = CartesianIndices(
            ntuple(
                β ->
                    α == β ?
                    iseven(comp) ? (div(comp, 2):div(comp, 2)+comp) :
                    (div(comp, 2)+1:div(comp, 2)+comp) : (1:comp),
                D,
            ),
        )
        Φ!(backend, workgroupsize)(v, u, Val(α), volume, I0; ndrange)
    end
    v
end

function lesdatagen(dnsobs, Φ, les, compression, to_setup, n_plot)
    #p = scalarfield(les)
    Φu = vectorfield(les)

    #results = (; u = fill(Array.(dnsobs[].u), 0), c = fill(Array.(dnsobs[].u), 0))
    results = (; u = fill(Array(Φu), 0), qoi_hist = fill(zeros(typeof(les.Re),0), 0))
    on(dnsobs) do (; u, t, n)
        Φ(Φu, u, les, compression)
        apply_bc_u!(Φu, t, les)
        u_hat = get_u_hat(Φu, les, to_setup)
        w_hat = get_w_hat_from_u_hat(u_hat, to_setup)
        q = compute_QoI(u_hat, w_hat, to_setup,les)
        push!(results.qoi_hist, Array(q))

        n % n_plot == 0 || return
        push!(results.u, Array(Φu))
        
    end
    results
end

"""
Save filtered DNS data.
"""
filtersaver(dns, les, filters, compression, to_setup_les; nupdate = 1, n_plot = 1000, checkpoints = nothing, checkpoint_name=nothing) =
    processor(
        (results, state) -> (; results..., comptime = time() - results.comptime),
    ) do state
        comptime = time()
        (; x) = dns.grid
        T = eltype(x[1])

        dnsobs = Observable((; state[].u, state[].t, state[].n))
        data = [
            lesdatagen(dnsobs, Φ, les[i], compression[i], to_setup_les[i], n_plot) for
            i = 1:length(les), Φ in filters
        ]
        results = (; data, comptime)
        #temp = nothing
        on(state) do (; u, t, n)
            if n % nupdate == 0
                dnsobs[] = (; u, t, n)
            end
            if !isnothing(checkpoints) && n in checkpoints
                filename = "$checkpoint_name/checkpoint_n$(n).jld2"
                u_cpu = Array(u)
                jldsave(filename; results, u_cpu)
            end 
        end
        state[] = state[] # Save initial conditions
        results
    end

checkpointer(checkpoints, file_name) =
    processor() do state
        on(state) do (; u, t, n)
            if n in checkpoints
                println("Update checkpoint at n = $n, t = $t, filename = $file_name")
                filename = "$(file_name).jld2"
                u_cpu = Array(u)
                jldsave(filename; u_cpu)
            end
        end
        1
    end