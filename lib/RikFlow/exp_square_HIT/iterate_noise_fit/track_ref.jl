if false                                               #src
    include("../src/RikFlow.jl")                  #src
    #include("../NeuralClosure/src/NeuralClosure.jl")   #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
    using .SymmetryClosure                             #src
    #using .NeuralClosure                               #src
    using .IncompressibleNavierStokes                  #src
end

using Random
using CairoMakie
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA
using RegularizedLeastSquares
using Distributions
using LinearAlgebra


ArrayType = CuArray
backend = CUDABackend()

#parse input ARGS
model_index = parse(Int, ARGS[1])
#model_index = 2
## Load data
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
(; name, hist_len, hist_var, n_replicas, normalization, include_predictor, train_range, indep_normals, lambda, fitted_qois) = inputs[model_index]
outdir = @__DIR__()*"/output/$(name)/"
ispath(outdir) || mkpath(outdir)

n_dns = Int(512)
n_les = Int(64)
Re = Float32(2_000)
############################
Δt = Float32(2.5e-3)
tsim = Float32(10)

# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber  
freeze = 1 # number of time steps to freeze the forcing

seeds = (;
    dns = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)


# For running on a CUDA compatible GPU
T = Float32

function fit_model(inputs, outputs, fitted_qois; indep_normals = false, lambda = 0.0, out_scaling = nothing)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term
    if lambda > 0.0
        reg = L2Regularization(lambda)
        solver = createLinearSolver(CGNR, inp; reg=reg)
        c = solve!(solver, outputs[fitted_qois,:]')
    else
        c = inp \ outputs[fitted_qois,:]'
    end 
    #For rectangular A the result is the minimum-norm least squares solution computed by a pivoted QR factorization of A and a rank estimate of A based on the R factor
    preds = inp * c
    stoch_part = copy(outputs)
    stoch_part[fitted_qois,:] -= preds'

    if !isnothing(out_scaling)
        stoch_part = RikFlow.scale_output(stoch_part, out_scaling)
    end
    # fit MVG
    if indep_normals
        stoch_distr = fit(DiagNormal, stoch_part .|> Float64)
    else
        stoch_distr = fit(MvNormal, stoch_part .|> Float64)
    end
    
    return c, stoch_distr
end

function create_history(hist_len, q_star, q, dQ; include_predictor = true)
    if hist_len == 0
        return q_star, dQ
    end
    qs = [q[:,hist_len-i+1:end-i+1] for i in 1:hist_len]
    if include_predictor
        return vcat(q_star[:,hist_len:end], qs...), dQ[:,hist_len:end]
    else
        return vcat(qs...), dQ[:,hist_len:end]
    end
end

function create_history(hist_len, q_star, q, dQ, hist_var; include_predictor = true)
    if hist_var == :q
        inputs,outputs = create_history(hist_len, q_star[:,:], q[:,:], dQ[:,:]; include_predictor)
    elseif hist_var == :q_star
        inputs,outputs = create_history(hist_len, q_star[:,2:end], q_star[:,1:end-1], dQ[:,2:end]; include_predictor)
    elseif hist_var == :q_star_q
        inputs,outputs = create_history(hist_len, q_star[:,2:end], cat(q[:,2:end],q_star[:,1:end-1],dims = 1), dQ[:,2:end]; include_predictor)
    end
    return inputs,outputs
end

# load reference data
ref_file = @__DIR__()*"/../output/new/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_freeze_10_tsim100.0.jld2"
data_train = load(ref_file, "data_train");
params_train = load(ref_file, "params_train");
# get initial condition
if data_train.data[1].u[1] isa Tuple
    ustart = stack(ArrayType.(data_train.data[1].u[1]));
elseif data_train.data[1].u[1] isa Array{<:Number,4}
    ustart = ArrayType(data_train.data[1].u[1]);
end

# get ref trajectories

tracking_noise = 0.0
n_iters = 20
noise_distrs = []
data_fractions = [0.2, 1.0]
conv_creteria = [0.2,0.1]
data = nothing
for (data_fraction, conv_c) in zip(data_fractions, conv_creteria)
    for i in 1:n_iters
        
        qoi_ref = stack(data_train.data[1].qoi_hist[1:Int(round(data_fraction*tsim/Δt))+1]);
        ref_reader = Reference_reader(qoi_ref);
        params_track = (;
            params_train...,
            tsim = round(data_fraction*tsim),
            Δt,
            ArrayType,
            backend,
            ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou),
            savefreq = 100,
            tracking_noise);

        data = track_ref(; params_track..., ref_reader, ustart);

        # fit linreg
        q_scaled, in_scaling = RikFlow._normalise(data.q[:,train_range[1]:end-1], normalization = normalization)
        q_star_scaled = RikFlow.scale_input(data.q_star[:,train_range[1]:end], in_scaling)
        dQ_scaled, out_scaling = RikFlow._normalise(data.dQ[:,train_range[1]:end], normalization = normalization)
        scaling = (;in_scaling, out_scaling)
        inputs, outputs = create_history(hist_len, q_star_scaled, q_scaled, dQ_scaled, hist_var; include_predictor)
        c, tracking_noise = fit_model(inputs, outputs, fitted_qois; indep_normals, lambda, out_scaling)
        push!(noise_distrs, tracking_noise)
        # check for convergence
        println("tracking_noise.Σ $(tracking_noise.Σ[1,1]) $(tracking_noise.Σ[2,2]) $(tracking_noise.Σ[3,3]) $(tracking_noise.Σ[4,4]) $(tracking_noise.Σ[5,5]) $(tracking_noise.Σ[6,6])")
        if size(noise_distrs,1) > 1
            #rel_error = norm(noise_distrs[end].Σ - noise_distrs[end-1].Σ)/norm(noise_distrs[end-1].Σ)
            rel_error = maximum(abs.(diag(noise_distrs[end].Σ) - diag(noise_distrs[end-1].Σ))./diag(noise_distrs[end-1].Σ))
            println("rel_error $(rel_error)")
            if rel_error < conv_c
                break
            end
        end
    end
end


data_track = data;
# Save tracking data
jldsave("$outdir/data_track_trackingnoise_Re$(Re)_tsim$(tsim).jld2"; data_track, params_track);
exit()
# plot convergence of Σ[1,1]
n = length(noise_distrs)
lines(1:n, [noise_distrs[i].Σ[1,1] for i in 1:n])
lines(1:n, [noise_distrs[i].Σ[2,2] for i in 1:n])
lines(1:n, [noise_distrs[i].Σ[3,3] for i in 1:n])

