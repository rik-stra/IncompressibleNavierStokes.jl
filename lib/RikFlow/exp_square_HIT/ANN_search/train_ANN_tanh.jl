if false
    include("../src/RikFlow.jl")
    using .RikFlow
end

using RikFlow
using MLUtils
using JLD2
using Lux, LuxCUDA
using Random

using Optimisers, Zygote, Printf, Statistics
const dev = gpu_device()
const cpu = cpu_device()

function create_history(hist_len, q_star, q, dQ)
    if hist_len == 0
        return q_star, dQ
    end
    qs = [q[:,hist_len-i+1:end-i+1] for i in 1:hist_len]
    return vcat(q_star[:,hist_len:end], qs...), dQ[:,hist_len:end]
end

#parse input ARGS
model_index = parse(Int, ARGS[1])

## Load data
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
(; name, track_file, hist_len, hist_var, lr, lambda, n_replicas, hidden_size, n_layers, batchsize, normalization) = inputs[model_index]

out_dir = @__DIR__()*"/output/$(name)/"
if isdir(out_dir)
    error("Output directory already exists: $(out_dir)")
else
    mkpath(out_dir)
    save(out_dir*"parameters.jld2", "parameters", (; name, track_file, hist_len, lr, lambda, n_replicas, hidden_size, n_layers, batchsize, normalization))
end

data = load(@__DIR__()*track_file, "data_track");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]

loss_function = Lux.MSELoss()
if hist_var == :q
    inputs,outputs = create_history(hist_len, data.q_star[:,1:3000], data.q[:,1:3000], data.dQ[:,1:3000])
elseif hist_var == :q_star
    inputs,outputs = create_history(hist_len, data.q_star[:,2:3000], data.q_star[:,1:3000-1], data.dQ[:,2:3000])
end

for k in 1:n_replicas
    rng = Random.Xoshiro(13+k)
    data_loaders, scaling=create_dataloader(inputs, outputs, batchsize, rng; normalization);
    model, ps, st, model_dict = setup_ANN(6, hist_len, 6, hidden_size, n_layers, rng, activation_function = tanh);
    @info model
    tstate = Lux.Training.TrainState(model, ps, st, AdamW(; eta=lr, lambda))
    tstate, losses = train_model(tstate, data_loaders, loss_function, 3000);

    fname = out_dir*"ANN_repl$(k).jld2"
    save(fname, "model_dict", model_dict, "parameters", tstate.parameters |> cpu, "states", tstate.states|> cpu, "scaling", scaling, "losses", losses)
end
