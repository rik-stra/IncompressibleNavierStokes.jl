module RikFlow
#= Implementation of tau-orthogonal method

=#
using IncompressibleNavierStokes
using FFTW
using Observables
using JLD2
using Infiltrator
using TensorOperations
import cuTENSOR

"""
Create setup for Tau-orthogonal method (stored in a named tuple).
"""
function TO_Setup(; qois, qoi_refs_folder, to_mode, ArrayType, setup, nstep)
    masks, ∂ = get_masks(qois, setup, ArrayType)
    N_qois = length(qois)
    
    to_setup = (; N_qois, qois, qoi_refs_folder, to_mode, masks, ∂)

    if to_mode == "TRACK_REF"
        qoi_trajectories = load(qoi_refs_folder*"QoIhist.jld2")["q"]
        time_index = ones(Int)
        time_index[] = 1
        V_i = get_vi_functions(to_setup)
        cij_masks = get_cij_masks(to_setup)
        outputs = allocate_arrays_outputs(nstep, N_qois)
        #P_hat, sgs_hat = allocate_arrays_to(to_setup, setup, ArrayType)

        to_setup = (; to_setup..., qoi_ref=(;qoi_trajectories,time_index), V_i, cij_masks, outputs)
    end
    return to_setup
end

function allocate_arrays_outputs(nstep, N_qois)
    dQ = Array{Float64}(undef, nstep, N_qois)
    tau = Array{Float64}(undef, nstep, N_qois)
    (; dQ, tau)
end

function allocate_arrays_to(to_setup, setup, ArrayType)
    (; Nu) = setup.grid
    (; N_qois) = to_setup
    d = length(Nu)
    P_hat = ArrayType{ComplexF64}(undef, Nu[1][1], Nu[1][2], Nu[1][3], d, N_qois)
    sgs_hat = ArrayType{ComplexF64}(undef, Nu[1][1], Nu[1][2], Nu[1][3], d)
    return P_hat, sgs_hat
end

function get_cij_masks(to_setup)
    mask_A = ones(Bool,(to_setup.N_qois, to_setup.N_qois, to_setup.N_qois))
    mask_B = ones(Bool,(to_setup.N_qois, to_setup.N_qois))
    for i in 1:to_setup.N_qois
        mask_A[:,i,i] .= false
        mask_A[i,:,i] .= false
        mask_B[i,i] = false
    end
    return (A=mask_A, B=mask_B)
end

function get_vi_functions(to_setup)
    vi = []
    for i in 1:to_setup.N_qois
        if to_setup.qois[i][1] == "E"
            f = (u,w) -> to_setup.masks[i].*u
        elseif to_setup.qois[i][1] == "Z"
            f = (u,w) -> 2*curl(to_setup.masks[i].*w, to_setup)
            #f = (u,w) -> to_setup.masks[i].*w
        end
        push!(vi, f)
    end
    return vi
end

function curl(x, to_setup)
    (; ∂) = to_setup
    return stack(
        (
            ∂[2].*x[:,:,:,3] .- ∂[3].*x[:,:,:,2],
            ∂[3].*x[:,:,:,1] .- ∂[1].*x[:,:,:,3],
            ∂[1].*x[:,:,:,2] .- ∂[2].*x[:,:,:,1],
        ),
        dims = 4
    )
end

function get_masks(QoIs, setup, ArrayType)
    N = setup.grid.Np
    xlims = setup.grid.xlims
    k = fftfreq(N[1], N[1])
    l = fftfreq(N[2], N[2])
    m = fftfreq(N[3], N[3])
    # create a list of bolean arrays
    masks_list = [Array{Bool, length(N)}(undef,N) for i in 1:length(QoIs)]
    #println("masks_list: ", typeof(masks_list))
    for q in 1:length(QoIs)
        for r in 1:N[3], j in 1:N[2], i in 1:N[1]
            if (k[i]^2 + l[j]^2 + m[r]^2) >= QoIs[q][2] && (k[i]^2 + l[j]^2 + m[r]^2) <= QoIs[q][3]
                masks_list[q][i,j,r] = true
            else
                masks_list[q][i,j,r] = false
            end
        end
    end
    masks_list = ArrayType.(masks_list)
    ∂ = [2*pi/(xlims[1][2] - xlims[1][1]).*reshape(l,(:,1,1))*1im,
    2*pi/(xlims[2][2] - xlims[2][1]).*reshape(l,(1,:,1))*1im,
    2*pi/(xlims[3][2] - xlims[3][1]).*reshape(l,(1,1,:))*1im]
    ∂ = ArrayType.(∂)
    return masks_list, ∂
end

function get_qoi_ref(to_setup)
    (; qoi_trajectories, time_index) = to_setup.qoi_ref
    q = qoi_trajectories[:,time_index[]]
    time_index[] += 1
    return q
end

function compute_QoI(u_hat, w_hat, to_setup, setup)
    (; dimension, xlims) = setup.grid
    L = xlims[1][2] - xlims[1][1]
    @assert (xlims[2][2] - xlims[2][1] == L) && (xlims[3][2] - xlims[3][1] == L)
    
    D = dimension()
    N = size(u_hat, 1)
    q = zeros(to_setup.N_qois)

    # E = real(sum(u_hat.*conj(u_hat),dims = 4))
    # Z = real(sum(w_hat.*conj(w_hat),dims = 4))
    E = sum(abs2, u_hat, dims = 4)
    Z = sum(abs2, w_hat, dims = 4)
    for i in 1:to_setup.N_qois
        if to_setup.qois[i][1] == "E"
            q[i]= sum(E.*to_setup.masks[i])*(L^D*1/(2*N^(2*D)))
        elseif to_setup.qois[i][1] == "Z"
            q[i] = sum(Z.*to_setup.masks[i])*(L^D*1/(N^(2*D)))
        else
            error("QoI not recognized")
        end 
    end
    return q

end


"""
    get_u_hat(u::Tuple, setup)
Compute the Fourier transform of the field. Returns an 4D array, velocity components stacked along last dimension.
"""
function get_u_hat(u::Tuple, setup)
    (; dimension, Iu) = setup.grid
    d = dimension()
    # interpolate u to cell centers
    #u_c = interpolate_u_p(u, setup)
    u = stack([u[a][Iu[a]] for a=1:d], dims=4)
    u_hat = fft(u, [1,2,3])
    return u_hat
end

"""
    get_w_hat(u::Tuple, setup)
Compute the vorticity field, interpolate to cell centers and compute the Fourier transform of the field.
"""
function get_w_hat(u::Tuple, setup)
    (; Ip) = setup.grid
    # compute vorticity
    w = vorticity(u, setup)
    # interpolate w to cell centers
    w = interpolate_ω_p(w, setup) 
    w = stack(w, dims=4)[Ip,:]
    # compute Fourier transform
    w_hat = fft(w, [1,2,3])
    return w_hat
end

function get_w_hat_from_u_hat(u_hat, to_setup)
    
    # compute vorticity
    w_hat = curl(u_hat, to_setup)
    return w_hat
end

using IncompressibleNavierStokes: timestep!, create_stepper, get_state, default_psolver, 
    ode_method_cache, AbstractODEMethod, AbstractRungeKuttaMethod, RKMethods, processor, apply_bc_u!

"""
Create processor that stores the QoI values every `nupdate` time step.
"""
qoisaver(; setup, to_setup, nupdate = 1) =
    processor() do state
        qoi_hist = fill(zeros(0), 0)
        on(state) do state
            state.n % nupdate == 0 || return
            u_hat = get_u_hat(state.u, setup)
            w_hat = get_w_hat_from_u_hat(u_hat, to_setup)
            q = compute_QoI(u_hat, w_hat, to_setup,setup)
            push!(qoi_hist, q)
        end
        state[] = state[]
        qoi_hist
    end

struct TOMethod{T,R,TOS} <: AbstractODEMethod{T}
    rk_method::R
    to_setup::TOS
    TOMethod(; rk_method = RKMethods.RK44(), to_setup) = new{eltype(rk_method.A),typeof(rk_method),typeof(to_setup)}(rk_method, to_setup)
end

export TOMethod

IncompressibleNavierStokes.create_stepper(method::TOMethod; setup, psolver, u, temp, t, n = 0) =
    create_stepper(method.rk_method; setup, psolver, u, temp, t, n)

IncompressibleNavierStokes.ode_method_cache(method::TOMethod, setup, u, temp) =
 ode_method_cache(method.rk_method, setup, u, temp)

function IncompressibleNavierStokes.timestep!(method::TOMethod, stepper, Δt; θ = nothing, cache)
    (; rk_method, to_setup) = method
    (; setup) = stepper
    (; dimension, Iu) = setup.grid
    D = dimension()

    stepper = timestep!(method.rk_method, stepper, Δt; θ, cache)

    # to method
    sgs = to_sgs_term(stepper.u, setup, to_setup, stepper)
    
    for a in 1:D
        stepper.u[a][Iu[a]] .+= sgs[:,:,:,a]
    end

    apply_bc_u!(stepper.u, stepper.t, setup)
    stepper
end

function to_sgs_term(u, setup, to_setup, stepper)
    N = setup.grid.Np[1]
    D = setup.grid.dimension()
    # get u_hat v_hat
    u_hat = get_u_hat(u, setup);
    w_hat = get_w_hat_from_u_hat(u_hat, to_setup);
    #w_hat = get_w_hat(u, setup);
    
    # get dQ
    q = compute_QoI(u_hat, w_hat, to_setup,setup)
    q_ref = get_qoi_ref(to_setup)
    dQ = q_ref-q
    to_setup.outputs.dQ[stepper.n,:] = dQ
    
    # get V_i
    vi = [to_setup.V_i[i](u_hat, w_hat) for i in 1:to_setup.N_qois]
    vi = stack(vi, dims=5)

    # get T_i
    ti = copy(vi)
    # compute innerproducts
    ip = innerpoducts(vi,ti,N,D)
    
    # compute c_ij
    cij = compute_cij(ip, to_setup)
    
   
    src_Q = reshape(sum(-conj(cij).*ip, dims = 1),:)
    #println("src_Q: ", src_Q)
    tau = dQ./src_Q
    to_setup.outputs.tau[stepper.n,:] = real(tau)
    #println("tau ", tau)

    # move to GPU
    cij = setup.ArrayType(cij)
    tau = setup.ArrayType(tau)
    # construct SGS term
    
    @tensor P_hat[c,d,e,f,b] := cij[a,b]* ti[c,d,e,f,a]
    @tensor sgs_hat[b,c,d,e] := -tau[a] * P_hat[b,c,d,e,a]
    sgs = real(ifft(sgs_hat, [1,2,3]))
    
#    u_new = u_hat+sgs_hat
#    w_new = curl(u_new, to_setup)
#    q_new = compute_QoI(u_new, w_new, to_setup,setup)

#    println("q_ref  = ", q_ref)
#    println("q      = ", q)
#    println("q_new  = ", q_new)
    

    return sgs
end

function innerpoducts(x,y,N,D)
    ip = reshape(
        sum(
            x.*conj(reshape(y, (size(y)[1:end-1]..., 1, size(y)[end]))),
             dims = (1,2,3,4)
        ),
        (size(y)[end],size(y)[end]))./N^(2*D)
    Array(ip)
end

function compute_cij(ip, to_setup) 
    N = to_setup.N_qois
    cij = ones(ComplexF64, (N, N)).*-1
    for i in 1:N
        A = reshape(ip[to_setup.cij_masks.A[:,:,i]], (N-1, N-1))
        b = ip[:,i][to_setup.cij_masks.B[:,i]]
        cij[to_setup.cij_masks.B[:,i],i] = A\b  # each column of cij is the solution of a linear system
    end
    return cij
end



end # module RikFlow