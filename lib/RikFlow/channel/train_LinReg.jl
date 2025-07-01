if false
    include("../src/RikFlow.jl")
    using .RikFlow
end

using RikFlow
using JLD2
using Random
using CairoMakie
using Distributions
using LinearAlgebra
using RegularizedLeastSquares
 
#parse input ARGS
model_index = parse(Int, ARGS[1])
#model_index = 6

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


## Load data
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
(; name, hist_len, hist_var, n_replicas, normalization, include_predictor, tracking_noise, train_range, indep_normals, lambda, fitted_qois, model_noise) = inputs[model_index]


out_dir = @__DIR__()*"/output/online_TOpaper/$(name)/"
if !isdir(out_dir)
    mkpath(out_dir)
end
save(out_dir*"parameters.jld2", "parameters", (; name, hist_len, hist_var, n_replicas, normalization, include_predictor))

track_file = @__DIR__()*"/output/track/LF_6qoinew_mirror_track_channel_to_64_64_32_dt0.005_tsim10.0.jld2"
data = load(track_file, "data_train");


q_scaled, in_scaling = RikFlow._normalise(data.q[:,train_range[1]:train_range[2]-1], normalization = normalization)
q_star_scaled = RikFlow.scale_input(data.q_star[:,train_range[1]:train_range[2]-1], in_scaling)
dQ_scaled     = RikFlow.scale_input(data.q[:,train_range[1]+1:train_range[2]], in_scaling)
scaling = (in_scaling = in_scaling, out_scaling = in_scaling)

inputs, outputs = create_history(hist_len, q_star_scaled, q_scaled, dQ_scaled, hist_var; include_predictor)


function fit_model(inputs, outputs, fitted_qois; indep_normals = false, lambda = 0.0, regularizer = :l2)
    n_targets = length(fitted_qois)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term
    if lambda > 0.0
        inp_r = kron(Matrix(I, n_targets,n_targets),inp)
        if regularizer == :l2
            reg = L2Regularization(lambda)
        elseif regularizer == :nuclear
            reg = NuclearRegularization(lambda, (size(inp,2), size(outputs,1)))
        end
        solver = createLinearSolver(ADMM, inp_r; reg=reg)
        b = reshape(outputs[fitted_qois,:]', length(fitted_qois)*size(outputs,2),1)
        c = solve!(solver, b)
        c = reshape(c, :, length(fitted_qois))
    else
        c = inp \ outputs[fitted_qois,:]'
    end 
    
    preds = inp * c
    stoch_part = copy(outputs)
    stoch_part[fitted_qois,:] -= preds'

    #loss  = norm(preds)
    # fit MVG
    if indep_normals
        stoch_distr = fit(DiagNormal, stoch_part .|> Float64)
    else
        stoch_distr = fit(MvNormal, stoch_part .|> Float64)
    end
    
    return c, stoch_distr
end

function run_model(inputs, c, stoch_distr, fitted_qois)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term
    preds = inp * c
    rand_part = rand(stoch_distr, size(inputs,2))'
    return rand_part[fitted_qois,:]+=preds
end

# fit model
c, stoch_distr = fit_model(inputs, outputs, fitted_qois; indep_normals, lambda, regularizer = :l2)

# use same noise distr as in tracking
if model_noise == :tracking_noise
    stds_ref_data = load(@__DIR__()*"/output/tracking/stds_refdata.jld2", "stds")
    stds = stds_ref_data.*tracking_noise./scaling.out_scaling.sigma
    stoch_distr = MvNormal(diagm(reshape(stds,6).^2))
elseif model_noise == :no_noise
    stoch_distr = nothing
end

## save model
save(out_dir*"/LinReg.jld2", "c", c', "stoch_distr", stoch_distr, 
    "scaling", scaling, "hist_var", hist_var, "hist_len", hist_len, "include_predictor", include_predictor, "fitted_qois", fitted_qois)
exit()

function plot_time_series(data, qois, title; ref = nothing)
    g = Figure(size = (800, 800))
    axs = [Axis(g[2+(i ÷ 2), i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(data, 1)-1]
    for i in 1:size(data, 1)
        lines!(axs[i], data[i,:])
        if ref != nothing
            lines!(axs[i], ref[i,:], color = :black)
        end
        
        
    end
    g[1,:] = Label(g, title, fontsize = 24, color = :blue)
    display(g)
end

## test the model
qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10],["Z",11,17],["E", 11, 17]];
track_file = @__DIR__()*"/output/track/LF_6qoinew_mirror_track_channel_to_64_64_32_dt0.005_tsim10.0.jld2"
data_test = load(track_file, "data_train");

dir = @__DIR__()*"/output/online_TOnew/LinReg6/"
model = load(dir*"LinReg.jld2")
hist_var = model["hist_var"]
include_predictor = model["include_predictor"]

# scale inputs and outputs
q_test = RikFlow.scale_input(data_test.q[:,1:2000], model["scaling"].in_scaling)
q_star_test = RikFlow.scale_input(data_test.q_star[:,1:2000], model["scaling"].in_scaling)
dQ_test = RikFlow.scale_input(data_test.q[:,2:2001], model["scaling"].out_scaling)
dQ_scaled = data_test.dQ[:,1:2000]./ model["scaling"].out_scaling.sigma

inputs_test,outputs_test = create_history(model["hist_len"], q_star_test, q_test, dQ_test, hist_var; include_predictor)


inp = cat(inputs_test',ones(eltype(inputs_test), (size(inputs_test,2),1)),dims=2)
rng = Xoshiro(12)
rand_part = rand(rng, model["stoch_distr"], size(inputs_test,2))'
rp = copy(rand_part)
rand_unsc = RikFlow.scale_output(rand_part', model["scaling"].out_scaling)

preds = rand_part'
preds[fitted_qois,:] += model["c"] * inp'
preds_unsc = RikFlow.scale_output(preds, model["scaling"].out_scaling)

plot_time_series(preds, qois, "preds", ref = outputs_test)
plot_time_series(preds-q_star_test[:,6:end], qois, "preds", ref = dQ_scaled[:,:])
plot_time_series(rp', qois, "rand_part", ref=tracking_noise.*randn(6,1000))

# plot original time series

plot_time_series(preds_unsc, qois, "preds_unsc", ref = data_test.q[:,model["hist_len"]+1:1000])
