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

function create_history(hist_len, q_star, q, dQ)
    if hist_len == 0
        return q_star, dQ
    end
    qs = [q[:,hist_len-i+1:end-i+1] for i in 1:hist_len]
    return vcat(q_star[:,hist_len:end], qs...), dQ[:,hist_len:end]
end

#parse input ARGS
model_index = parse(Int, ARGS[1])
model_index =7
## Load data
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
(; name, track_file, hist_len, hist_var, n_replicas, normalization) = inputs[model_index]

out_dir = @__DIR__()*"/output/$(name)/"
if isdir(out_dir)
    error("Output directory already exists: $(out_dir)")
else
    mkpath(out_dir)
    save(out_dir*"parameters.jld2", "parameters", (; name, track_file, hist_len, hist_var, n_replicas, normalization))
end


data = load(@__DIR__()*track_file, "data_track");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
n_qois = length(qois)


if hist_var == :q
    inputs,outputs = create_history(hist_len, data.q_star[:,1:3000], data.q[:,1:3000], data.dQ[:,1:3000])
elseif hist_var == :q_star
    inputs,outputs = create_history(hist_len, data.q_star[:,2:3000], data.q_star[:,1:3000-1], data.dQ[:,2:3000])
end

# normalize inputs and outputs
inputs_scaled, in_scaling = RikFlow._normalise(inputs, normalization = normalization)
outputs_scaled, out_scaling = RikFlow._normalise(outputs, normalization = normalization)
scaling = (;in_scaling, out_scaling)

inp = cat(inputs_scaled',ones(eltype(inputs_scaled), (size(inputs_scaled,2),1)),dims=2) # add a bias term
c = inp \ outputs_scaled' 
#For rectangular A the result is the minimum-norm least squares solution computed by a pivoted QR factorization of A and a rank estimate of A based on the R factor

preds = inp * c
stoch_part = outputs_scaled - preds'
loss  = norm(preds)
# fit MVG
stoch_distr = fit(MvNormal, stoch_part .|> Float64)

## save model
save(out_dir*"/LinReg.jld2", "c", c', "stoch_distr", stoch_distr, "scaling", scaling, "hist_var", hist_var, "hist_len", hist_len)
exit()


## test the model
dir = @__DIR__()*"/output/LinReg7/"
model = load(dir*"LinReg.jld2")
hist_var = model["hist_var"]
#hist_var = :q
if hist_var == :q
    inputs_test,outputs_test = create_history(model["hist_len"], data.q_star[:,1:4000], data.q[:,1:4000], data.dQ[:,1:4000])
elseif hist_var == :q_star
    inputs_test,outputs_test = create_history(model["hist_len"], data.q_star[:,2:4000], data.q_star[:,1:4000-1], data.dQ[:,2:4000])
end
inputs_test_sc = RikFlow.scale_input(inputs_test, model["scaling"].in_scaling)
outputs_test_sc = RikFlow.scale_input(outputs_test, model["scaling"].out_scaling)


inp = cat(inputs_test_sc',ones(eltype(inputs_test_sc), (size(inputs_test_sc,2),1)),dims=2)
rng = Xoshiro(12)
rand_part = rand(rng, model["stoch_distr"], size(inputs_test_sc,2))'
preds = model["c"] * inp' + rand_part'


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

plot_time_series(preds, qois, "preds", ref = outputs_test_sc)
plot_time_series(rand_part', qois, "stoch_part", ref = stoch_part)

# plot original time series
preds_unsc = RikFlow.scale_output(preds', out_scaling)
plot_time_series(preds_unsc, qois, "preds_unsc", ref = outputs_test)