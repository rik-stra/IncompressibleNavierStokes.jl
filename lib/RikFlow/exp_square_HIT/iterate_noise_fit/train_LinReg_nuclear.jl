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
#model_index = parse(Int, ARGS[1])
model_index =7

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
(; name, hist_len, hist_var, n_replicas, normalization, include_predictor, train_range, tracking_noise, indep_normals, lambda, fitted_qois) = inputs[model_index]
norm = :l2
lambda = 0.001

out_dir = @__DIR__()*"/output/$(name)/"
save(out_dir*"parameters.jld2", "parameters", (; name, hist_len, hist_var, n_replicas, normalization, include_predictor))
track_file = out_dir*"data_track_trackingnoise_Re2000.0_tsim10.0.jld2"

data = load(track_file, "data_track");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]

q_scaled, in_scaling = RikFlow._normalise(data.q[:,train_range[1]:train_range[2]-1], normalization = normalization)
q_star_scaled = RikFlow.scale_input(data.q_star[:,train_range[1]:train_range[2]-1], in_scaling)
dQ_scaled, out_scaling = RikFlow._normalise(data.q[:,train_range[1]+1:train_range[2]], normalization = normalization)
scaling = (;in_scaling, out_scaling)

inputs, outputs = create_history(hist_len, q_star_scaled, q_scaled, dQ_scaled, hist_var; include_predictor)




function fit_model(inputs, outputs, fitted_qois; indep_normals = false, lambda = 0.0, norm = :l2)
    n_targets = length(fitted_qois)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term
    if lambda > 0.0
        if norm == :l2
            reg = L2Regularization(lambda)
            solver = createLinearSolver(CGNR, inp; reg=reg)
            c = solve!(solver, outputs[fitted_qois,:]')
        elseif norm == :nuclear
            inp_r = kron(Matrix(I, n_targets,n_targets),inp)
            reg = NuclearRegularization(lambda, (size(inp,2), size(outputs,1)))
            #reg = L2Regularization(lambda)
            solver = createLinearSolver(ADMM, inp_r; reg=reg)
            b = reshape(outputs[fitted_qois,:]', length(fitted_qois)*size(outputs,2),1)
            c = solve!(solver, b)
            c = reshape(c, :, length(fitted_qois))
        end
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
c, stoch_distr = fit_model(inputs, outputs, fitted_qois; indep_normals, lambda, norm)

inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2)
x = c' * inp'

outputs

# g = Figure();
# ax,hm = heatmap(g[1,1], c, 
# #colormap = :grays, colorrange = (-5, 5), highclip = :red, lowclip = :blue)
# colormap = :balance, colorrange = (-25,25))
# Colorbar(g[1, 2], hm)
# Label(g[0,:], text = "lambda $(lambda)", fontsize = 20)
# display(g)

## save model
save(out_dir*"/LinReg_l2.jld2", "c", c', "stoch_distr", stoch_distr, 
    "scaling", scaling, "hist_var", hist_var, "hist_len", hist_len, "include_predictor", include_predictor, "fitted_qois", fitted_qois, "norm", norm)
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
data_test = load(@__DIR__()*"/../output/new/data_track2_dns512_les64_Re2000.0_tsim100.0.jld2", "data_track");
dir = @__DIR__()*"/output/LinReg4/"
model = load(dir*"LinReg_nuclear.jld2")
hist_var = model["hist_var"]
include_predictor = model["include_predictor"]

# scale inputs and outputs
q_test = RikFlow.scale_input(data_test.q[:,1:8000], model["scaling"].in_scaling)
q_star_test = RikFlow.scale_input(data_test.q_star[:,1:8000], model["scaling"].in_scaling)
dQ_test = RikFlow.scale_input(data_test.q[:,2:8001], model["scaling"].out_scaling)

inputs_test,outputs_test = create_history(model["hist_len"], q_star_test, q_test, dQ_test, hist_var; include_predictor)


inp = cat(inputs_test',ones(eltype(inputs_test), (size(inputs_test,2),1)),dims=2)
rng = Xoshiro(12)
rand_part = rand(rng, model["stoch_distr"], size(inputs_test,2))'
preds = rand_part'
preds[fitted_qois,:] += model["c"] * inp'

plot_time_series(preds, qois, "preds", ref = outputs_test)

# plot original time series
preds_unsc = RikFlow.scale_output(preds, out_scaling)
plot_time_series(preds_unsc, qois, "preds_unsc", ref = data_test.dQ[:,model["hist_len"]+1:8000])