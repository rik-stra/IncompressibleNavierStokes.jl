if false
    include("../src/RikFlow.jl")
    using .RikFlow
end

using RikFlow
using MLUtils
using JLD2
using CairoMakie
using Lux, LuxCUDA
using Random

using Optimisers, Zygote, Printf, Statistics
const dev = gpu_device()
const cpu = cpu_device()

function plot_time_series(data, qois, title; ref = nothing)
    g = Figure(size = (800, 800))
    axs = [Axis(g[2+(i ÷ 2), i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(data, 1)-1]
    for i in 1:size(data, 1)
        if ref != nothing
            lines!(axs[i], ref[i,:], color = :black)
        end
        lines!(axs[i], data[i,:])
        
    end
    g[1,:] = Label(g, title, fontsize = 24, color = :blue)
    display(g)
end

function create_history(hist_len, q_star, q, dQ)
    if hist_len == 0
        return q_star, dQ
    end
    qs = [q[:,hist_len-i+1:end-i+1] for i in 1:hist_len]
    return vcat(q_star[:,hist_len:end], qs...), dQ[:,hist_len:end]
end


rng = Random.Xoshiro(13)
## Load data
filename = @__DIR__()*"/../exp_square_HIT/output/new/data_track_qstar2_dns512_les64_Re2000.0_tsim10.0.jld2"
data = load(filename, "data_track");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]

hist_len = 3
inputs,outputs = create_history(hist_len, data.q_star[:,1:3000], data.q[:,1:3000], data.dQ[:,1:3000])


data_loaders, scaling=create_dataloader(inputs, outputs, 64, rng, normalization =:minmax);
model, ps, st, model_dict = setup_ANN(6, hist_len, 6, 64, 3, rng, activation_function = tanh);
model

lr = 0.005f0
loss_function = Lux.MSELoss()
tstate = Lux.Training.TrainState(model, ps, st, AdamW(eta=lr, lambda=0.1))
tstate = train_model(tstate, data_loaders, loss_function, 3000);


## Plot data
plot_time_series(data.q_star, qois, "q*")
plot_time_series(data.dQ, qois, "dQ")

## Predict
inputs,outputs = create_history(hist_len, data.q_star[:,:], data.q[:,1:end-1], data.dQ[:,:])
x = scale_input(inputs, scaling.in_scaling)
st_ = Lux.testmode(tstate.states)
y_pred = Lux.apply(tstate.model, dev(x), tstate.parameters, st_)[1]
y_pred = scale_output(Array(y_pred), scaling.out_scaling)
plot_time_series(Array(y_pred), qois, "Predicted dQ", ref = data.dQ)

## Save model
filename = @__DIR__()*"/output/trained_models/ANN_tanh_regularized_hist3.jld2"
save(filename, "model_dict", model_dict, "parameters", tstate.parameters |> cpu, "states", tstate.states|> cpu, "scaling", scaling)

