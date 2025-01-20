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

#parse input ARGS
#model_index = parse(Int, ARGS[1])
model_index =4
## Load data
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
(; name, track_file, hist_len, hist_var, n_replicas, normalization, include_predictor, train_range) = inputs[model_index]

out_dir = @__DIR__()*"/output/$(name)/"
if isdir(out_dir)
    error("Output directory already exists: $(out_dir)")
else
    mkpath(out_dir)
    save(out_dir*"parameters.jld2", "parameters", (; name, track_file, hist_len, hist_var, n_replicas, normalization, include_predictor))
end

data = load(@__DIR__()*track_file, "data_track");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]

q_scaled, in_scaling = RikFlow._normalise(data.q[:,train_range[1]:train_range[2]], normalization = normalization)
q_star_scaled = RikFlow.scale_input(data.q_star[:,train_range[1]:train_range[2]], in_scaling)
dQ_scaled, out_scaling = RikFlow._normalise(data.dQ[:,train_range[1]:train_range[2]], normalization = normalization)
scaling = (;in_scaling, out_scaling)

inputs, outputs = create_history(hist_len, q_star_scaled, q_scaled, dQ_scaled, hist_var; include_predictor)


function fit_model(inputs, outputs)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term
    c = inp \ outputs' 
    #For rectangular A the result is the minimum-norm least squares solution computed by a pivoted QR factorization of A and a rank estimate of A based on the R factor
    preds = inp * c
    stoch_part = outputs - preds'
    #loss  = norm(preds)
    # fit MVG
    stoch_distr = fit(MvNormal, stoch_part .|> Float64)
    return c, stoch_distr
end

function fit_model(inputs, outputs, c, stoch_distr, alpha=0.5)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term
    c1 = inp \ outputs' 
    #For rectangular A the result is the minimum-norm least squares solution computed by a pivoted QR factorization of A and a rank estimate of A based on the R factor
    c = alpha*c + (1-alpha)*c1
    preds = inp * c1
    stoch_part = outputs - preds'
    #loss  = norm(preds)
    # fit MVG
    stoch_distr1 = fit(MvNormal, stoch_part .|> Float64)
    μ = alpha*stoch_distr.μ .+ (1-alpha)*stoch_distr1.μ
    Σ = alpha.*stoch_distr.Σ .+ (1-alpha).*stoch_distr1.Σ
    #println("Σ: ", Σ)
    stoch_distr = MvNormal(μ,Σ)
    return c, stoch_distr
end

function fit_model(inputs, outputs, c, alpha=0.5)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term
    c1 = inp \ outputs' 
    #For rectangular A the result is the minimum-norm least squares solution computed by a pivoted QR factorization of A and a rank estimate of A based on the R factor
    c = alpha*c + (1-alpha)*c1
    preds = inp * c
    stoch_part = outputs - preds'
    #loss  = norm(preds)
    # fit MVG
    stoch_distr = fit(MvNormal, stoch_part .|> Float64)
    return c, stoch_distr
end

function run_model(inputs, c, stoch_distr)
    inp = cat(inputs',ones(eltype(inputs), (size(inputs,2),1)),dims=2) # add a bias term
    preds = inp * c
    rand_part = rand(stoch_distr, size(inputs,2))'
    return preds+rand_part
end

c_list = []
stoch_distr_list = []
# fit first model
c, stoch_distr = fit_model(inputs, outputs)
push!(c_list, c)
push!(stoch_distr_list, stoch_distr)

n_iters = 100
for i in 1:n_iters
    preds = run_model(inputs[:,:], c, stoch_distr)
    dQ_pred = RikFlow.scale_output(preds', out_scaling)
    q_pred = q_star_scaled[:,end-size(dQ_pred,2)+1:end] + RikFlow.scale_input(dQ_pred, in_scaling)

    inputs1,outputs1 = create_history(hist_len, q_star_scaled[:,end-size(q_pred,2)+1:end], q_pred, dQ_scaled[:,end-size(q_pred,2)+1:end], hist_var; include_predictor)
    c, stoch_distr_fake = fit_model(inputs1, outputs1,c, stoch_distr, 0.6)

    push!(c_list, c)
    push!(stoch_distr_list, stoch_distr)
end

#plot convergence of c
g = Figure()
ax = Axis(g[1,1], title = "c[1,1]")
lines!(ax, [c[4,4] for c in c_list])
display(g)

#plot convergence of stoch_distr
g = Figure()
ax = Axis(g[1,1], title = "stoch_distr.Σ[j,j]")
for j in 1:size(stoch_distr.Σ,2)
    lines!(ax, [stoch_distr.Σ[j,j] for stoch_distr in stoch_distr_list], label = "j = $j")
end
display(g)

g = Figure()
ax = Axis(g[1,1], title = "stoch_distr.mu")
for j in 1:size(stoch_distr.μ,1)
    lines!(ax, [stoch_distr.μ[j] for stoch_distr in stoch_distr_list])
end
display(g)

#test last model
preds = run_model(inputs[:,:], c_list[end], stoch_distr_list[end])
plot_time_series(preds', qois, "preds", ref = dQ_scaled[:,hist_len+1:end])

dQ_pred = RikFlow.scale_input(RikFlow.scale_output(preds', out_scaling), in_scaling)
q_pred = q_star_scaled[:,end-size(dQ_pred,2)+1:end] + dQ_pred
inputs1,outputs1 = create_history(hist_len, q_star_scaled[:,end-size(q_pred,2)+1:end], q_pred, dQ_scaled[:,end-size(q_pred,2)+1:end], hist_var; include_predictor)
preds_on_self = run_model(inputs1[:,:], c_list[end], stoch_distr_list[end])
plot_time_series(preds_on_self', qois, "preds_on_self", ref = dQ_scaled[:,hist_len+1:end])
println("preds / preds_on_self mean: ")
println(mean(preds, dims=1))
println(mean(preds_on_self, dims=1))

#test first model
preds = run_model(inputs[:,:], c_list[1], stoch_distr_list[1])
plot_time_series(preds', qois, "preds", ref = dQ_scaled[:,hist_len+1:end])

dQ_pred = RikFlow.scale_input(RikFlow.scale_output(preds', out_scaling), in_scaling)
q_pred = q_star_scaled[:,end-size(dQ_pred,2)+1:end] + dQ_pred
inputs1,outputs1 = create_history(hist_len, q_star_scaled[:,end-size(q_pred,2)+1:end], q_pred, dQ_scaled[:,end-size(q_pred,2)+1:end], hist_var; include_predictor)
preds_on_self = run_model(inputs1[:,:], c_list[1], stoch_distr_list[1])
plot_time_series(preds_on_self', qois, "preds_on_self", ref = dQ_scaled[:,hist_len+1:end])

println("preds / preds_on_self mean: ")
println(mean(preds, dims=1))
println(mean(preds_on_self, dims=1))
println(mean(dQ_scaled[:,hist_len+1:end], dims=2))


## save model
save(out_dir*"/LinReg.jld2", "c", c_list[1]', "stoch_distr", stoch_distr_list[1], "scaling", scaling, "hist_var", hist_var, "hist_len", hist_len, "include_predictor", include_predictor)
exit()


## test the model
dir = @__DIR__()*"/output/LinReg2/"
model = load(dir*"LinReg.jld2")
hist_var = model["hist_var"]
include_predictor = model["include_predictor"]
#hist_var = :q
# scale inputs and outputs
q_test = RikFlow.scale_input(data.q[:,1:8000], model["scaling"].in_scaling)
q_star_test = RikFlow.scale_input(data.q_star[:,1:8000], model["scaling"].in_scaling)
dQ_test = RikFlow.scale_input(data.dQ[:,1:8000], model["scaling"].out_scaling)

inputs_test,outputs_test = create_history(model["hist_len"], q_star_test, q_test, dQ_test, hist_var; include_predictor)


inp = cat(inputs_test',ones(eltype(inputs_test), (size(inputs_test,2),1)),dims=2)
rng = Xoshiro(12)
rand_part = rand(rng, model["stoch_distr"], size(inputs_test,2))'
preds = model["c"] * inp' + rand_part'

plot_time_series(preds, qois, "preds", ref = outputs_test)

dQ_pred2 = RikFlow.scale_input(RikFlow.scale_output(preds, out_scaling), in_scaling)
q_pred2 = q_star_test[:,end-size(dQ_pred2,2)+1:end] + dQ_pred2
inputs_self,outputs_self = create_history(hist_len, q_star_test[:,end-size(q_pred2,2)+1:end], q_pred2, dQ_test[:,end-size(q_pred2,2)+1:end], hist_var; include_predictor)
preds_on_self = run_model(inputs_self[:,:], model["c"][:,:]', model["stoch_distr"])
plot_time_series(preds_on_self', qois, "preds_on_self", ref = outputs_self)

plot_time_series(rand_part', qois, "stoch_part")

# plot original time series
preds_unsc = RikFlow.scale_output(preds, out_scaling)
plot_time_series(preds_unsc, qois, "preds_unsc", ref = data.dQ[:,model["hist_len"]+1:8000])