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

# parse input ARGS
model_index = parse(Int, ARGS[1])
# or set model_index manually
# model_index = 2
inputs_file_name = "/inputs.jld2"
TO_folder = @__DIR__()*"/output/TO_LRS"

track_file = @__DIR__()*"/output/LF/track/track_TG_64_Re_800.0_tsim20.0.jld2"

function create_history(hist_len, q_star, q, dQ, scaling; include_predictor = true)
    if hist_len == 0
        inp, target = q_star, dQ
    else
        qs = [q[:,hist_len-i+1:end-i+1] for i in 1:hist_len]
        if include_predictor
            inp, target = vcat(q_star[:,hist_len:end], qs...), dQ[:,hist_len:end]
        else
            inp, target = vcat(qs...), dQ[:,hist_len:end]
        end
    end
    # remove data points where any of q_star = 0
    inp = inp[:,reshape(all( abs.(q_star[:,max(hist_len,1):end].*scaling.in_scaling.sigma) .> 0.5e-2, dims=1),:)]
    target = target[:,reshape(all( abs.(q_star[:,max(hist_len,1):end].*scaling.in_scaling.sigma) .> 0.5e-2, dims=1),:)]
    return inp, target
end

function create_history(hist_len, q_star, q, dQ, hist_var, scaling; include_predictor = true)
    if hist_var == :q
        inputs,outputs = create_history(hist_len, q_star[:,:], q[:,:], dQ[:,:], scaling; include_predictor)
    elseif hist_var == :q_star
        inputs,outputs = create_history(hist_len, q_star[:,2:end], q_star[:,1:end-1], dQ[:,2:end], scaling; include_predictor)
    elseif hist_var == :q_star_q
        inputs,outputs = create_history(hist_len, q_star[:,2:end], cat(q[:,2:end],q_star[:,1:end-1],dims = 1), dQ[:,2:end], scaling; include_predictor)
    end
    return inputs,outputs
end


## Load parameters
inputs = load(TO_folder*inputs_file_name, "inputs")
(; name, hist_len, hist_var, n_replicas, normalization, include_predictor, tracking_noise, train_range, indep_normals, lambda, fitted_qois, model_noise) = inputs[model_index]


out_dir = TO_folder*"/$(name)/"
save(out_dir*"parameters.jld2", "parameters", (; name, hist_len, hist_var, n_replicas, normalization, include_predictor))


data = load(track_file, "data_train");

# normalize the data
q_scaled, in_scaling = RikFlow._normalise(data.q[:,train_range[1]:train_range[2]-1], normalization = normalization)
q_star_scaled = RikFlow.scale_input(data.q_star[:,train_range[1]:train_range[2]-1], in_scaling)
dQ_scaled     = RikFlow.scale_input(data.q[:,train_range[1]+1:train_range[2]], in_scaling)
#dQ_scaled     = RikFlow.scale_input(data.dQ[:,train_range[1]:train_range[2]-1], in_scaling)
scaling = (in_scaling = in_scaling, out_scaling = in_scaling)

inputs, outputs = create_history(hist_len, q_star_scaled, q_scaled, dQ_scaled, hist_var, scaling; include_predictor)

function fit_model(inputs, outputs, fitted_qois; indep_normals = false, lambda = 0.0, regularizer = :l2)
    n_targets = length(fitted_qois)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term

    # solve linear regression
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
    
    # fit the stochastic part to the residuals
    preds = inp * c
    stoch_part = copy(outputs)
    stoch_part[fitted_qois,:] -= preds'
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

# overwrite the nose distribution
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


data_test = load(track_file, "data_train");
#dir = @__DIR__()*"/output/online_TOnew/LinReg1/"
model = load(out_dir*"LinReg.jld2")
hist_var = model["hist_var"]
include_predictor = model["include_predictor"]
q_test = RikFlow.scale_input(data_test.q[:,1:400], model["scaling"].in_scaling)
q_star_test = RikFlow.scale_input(data_test.q_star[:,1:400], model["scaling"].in_scaling)
dQ_test = RikFlow.scale_input(data_test.q[:,2:401], model["scaling"].out_scaling)
#dQ_test = RikFlow.scale_input(data_test.dQ[:,1:400], model["scaling"].out_scaling)
dQ_scaled = data_test.dQ[:,1:400]./ model["scaling"].out_scaling.sigma

inputs_test,outputs_test = create_history(model["hist_len"], q_star_test, q_test, dQ_test, hist_var, model["scaling"]; include_predictor)


inp = cat(inputs_test',ones(eltype(inputs_test), (size(inputs_test,2),1)),dims=2)
rng = Xoshiro(12)
rand_part = rand(rng, model["stoch_distr"], size(inputs_test,2))'
rp = copy(rand_part)
rand_unsc = RikFlow.scale_output(rand_part', model["scaling"].out_scaling)

preds = rand_part'
preds[fitted_qois,:] += model["c"] * inp'
preds_unsc = RikFlow.scale_output(preds, model["scaling"].out_scaling)

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
qois = [["Z",0,1],["E", 0, 1],["Z",2,3],["E", 2, 3],["Z",4,5],["E", 4, 5]];
plot_time_series(preds, qois, "preds", ref = outputs_test)
plot_time_series(preds-q_star_test[:,2:end], qois, "preds", ref = dQ_scaled[:,6:end])
#plot_time_series(rp', qois, "rand_part", ref=tracking_noise.*randn(6,1000))