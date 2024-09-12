module RikFlow
#= Implementation of tau-orthogonal method
The main functions in this module produce the Reduced tau ortogonal closure term:
- "to_track()"    produces a closure term that tracks the reference QoI trajectories
- "to_predict()"  online prediction of the closure term based on training data, obtained from tracking
=#
using IncompressibleNavierStokes
using FFTW
using Observables

"""
Create setup for Tau-orthogonal method (stored in a named tuple).
"""
function TO_Setup(; qois, qoi_refs_folder, to_mode, ArrayType, setup)
    masks = get_masks(qois, setup, ArrayType)
    N_qois = length(qois)
    to_setup = (; N_qois, qois, qoi_refs_folder, to_mode, masks)
    return to_setup
end

function get_masks(QoIs, setup, ArrayType)
    N = setup.grid.Np
    k = fftfreq(N[1], N[1])
    l = fftfreq(N[2], N[2])
    m = fftfreq(N[3], N[3])
    # create a list of bolean arrays
    masks_list = [ArrayType{Bool, length(N)}(undef,N) for i in 1:length(QoIs)]
    for q in 1:length(QoIs)
        for r in 1:N[3], j in 1:N[2], i in 1:N[1]
            if (k[i]^2 + l[j]^2 + m[r]^2) >= QoIs[q][2] && (k[i]^2 + l[j]^2 + m[r]^2) <= QoIs[q][3]
                masks_list[q][i,j,r] = true
            else
                masks_list[q][i,j,r] = false
            end
        end
    end
    return masks_list
end

function to_track(u)
    (; grid) = setup
    (; dimension, Np, Ip) = grid
    D = dimension()
    # compute QoI

    # get QoI ref from file

    # get base functions T_i (divergence free)

    # get partial derivatives V_i

    # get O_i

    # compute closure term
    pass
end

function compute_QoI(u_hat, w_hat, to_setup, setup)
    (; dimension, xlims) = setup.grid
    L = xlims[1][2] - xlims[1][1]
    @assert (xlims[2][2] - xlims[2][1] == L) && (xlims[3][2] - xlims[3][1] == L)
    
    D = dimension()
    N = size(u_hat, 1)
    q = zeros(to_setup.N_qois)

    E = sum(u_hat.*conj(u_hat),dims = 4)
    Z = sum(w_hat.*conj(w_hat),dims = 4)
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
    apply_round_filter(u_hat::Array, bin::Tuple)
set all Fourier coefficients outside the wavenumber bin to zero. Only for 3D!
"""
function apply_round_filter(u_hat::Array, bin::Tuple)
    k = fftfreq(size(u_hat, 1),size(u_hat, 1))
    l = fftfreq(size(u_hat, 2),size(u_hat, 2))
    m = fftfreq(size(u_hat, 3),size(u_hat, 3))
    for i in 1:size(u_hat, 1)
        for j in 1:size(u_hat, 2)
            for k in 1:size(u_hat, 3)
                if (k[i]^2 + l[j]^2 + m[k]^2) > bin[2] || (k[i]^2 + l[j]^2 + m[k]^2) < bin[1]
                    u_hat[i,j,k] = 0
                end
            end
        end
    end
end


"""
    get_u_hat(u::Tuple, setup)
Interpolate the velocity field to cell centers and compute the Fourier transform of the field. Returns an 4D array, velocity components stacked along last dimension.
"""
function get_u_hat(u::Tuple, setup)
    (; Ip) = setup.grid
    # interpolate u to cell centers
    u_c = interpolate_u_p(u, setup)
    u_c = stack(u_c, dims=4)[Ip,:]
    u_hat = fft(u_c, [1,2,3])
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

"""
Create processor that stores the QoI values every `nupdate` time step.
"""
qoisaver(; setup, to_setup, nupdate = 1) =
    processor() do state
        qoi_hist = fill(zeros(0), 0)
        on(state) do state
            state.n % nupdate == 0 || return
            u_hat = get_u_hat(state.u, setup)
            w_hat = get_w_hat(state.u, setup)
            q = compute_QoI(u_hat, w_hat, to_setup,setup)
            push!(qoi_hist, q)
        end
        state[] = state[]
        qoi_hist
    end

end # module RikFlow
