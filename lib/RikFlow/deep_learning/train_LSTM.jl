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


function _normalise(x; normalization= :normal, dims=ndims(x), ϵ=1e-6)
    if normalization == :normal
        ϵ = convert(eltype(x), ϵ)
        mu = mean(x, dims=dims)
        sigma = std(x, dims=dims, corrected=false).+ ϵ
        
    elseif normalization == :minmax
        min = minimum(x, dims=dims)
        max = maximum(x, dims=dims)
        mu = 0.5(min+max)
        sigma = 0.5(max-min)
    end
    return (x .- mu) ./ (sigma), (;mu,sigma)
end

function extract_sequences(q_star, dQ, sequence_length)
    n_sequences = size(q_star, 2) ÷ (sequence_length ÷ 2) -1
    q_sequences = [q_star[:, (i-1)*sequence_length ÷ 2 + 1:(i-1)*sequence_length ÷ 2 +sequence_length] for i in 1:n_sequences]
    #dQ_sequences = [dQ[:, (i-1)*sequence_length ÷ 2 + 1:(i-1)*sequence_length ÷ 2 +sequence_length] for i in 1:n_sequences]
    dQ_sequences = [dQ[:, (i-1)*sequence_length ÷ 2 +sequence_length] for i in 1:n_sequences]
    return stack(q_sequences), stack(dQ_sequences)
end

function create_sequenceloaders(data_in, data_out, sequence_length , batchsize, rng; normalization = :normal)
    data_in_scaled, in_scaling = _normalise(data_in; normalization)
    data_out_scaled, out_scaling = _normalise(data_out; normalization)
    seq_in, seq_out = extract_sequences(data_in_scaled, data_out_scaled, sequence_length)
    @show size(seq_in), size(seq_out)
    (in_train, out_train),(in_val, out_val) = splitobs((seq_in, seq_out); at = 0.9, shuffle = true) # can not pass rng!
    val_loader = DataLoader(collect.((in_val, out_val)); batchsize, rng) |> dev
    train_loader = DataLoader(collect.((in_train, out_train)); batchsize, shuffle=true, rng) |> dev
    loaders = (;train_loader, val_loader)
    return loaders, (;in_scaling, out_scaling)
end

rng = Random.Xoshiro(13)
## Load data
filename = @__DIR__()*"/../exp_square_HIT/output/new/data_track_qstar2_dns512_les64_Re2000.0_tsim10.0.jld2"
data = load(filename, "data_track");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]

data_loaders, scaling=create_sequenceloaders(data.q_star[:,1:3000], data.dQ[:,1:3000], 100, 12, rng, normalization =:minmax);

# plot some sequences

    (x,y),_=iterate(data_loaders.train_loader)
    plot_time_series(Array(x[:,:,1]), qois, "q*")
    plot_time_series(Array(y[:,:,1]), qois, "dQ")



function LSTM_regressor_simple(in_dims, hidden_dims, out_dims)
    model = Chain(
        Recurrence(LSTMCell(in_dims => hidden_dims), return_sequence = true),
        Recurrence(LSTMCell(hidden_dims => out_dims), return_sequence = false)
    )
    return model
end

## loss functions
lossfn = MSELoss()

function compute_loss(model, ps, st, (x, y))
    ŷ, st_ = model(x, ps, st)
    loss = lossfn(ŷ, y)
    return loss, st_, (; y_pred=ŷ)
end



# Create the model

model = LSTM_regressor_simple(6, 40, 6)
rng = Xoshiro(0)
ps, st = Lux.setup(rng, model) |> dev

train_state = Training.TrainState(model, ps, st, Adam(0.01f0))

# one model pass
(x, y), _ = iterate(data_loaders.train_loader)
y_pred,_ = model(x, ps, st)

for epoch in 1:25
    # Train the model
    for (x, y) in data_loaders.train_loader
        (_, loss, _, train_state) = Training.single_train_step!(
            AutoZygote(), lossfn, (x, y), train_state)

        @printf "Epoch [%3d]: Loss %4.5f\n" epoch loss
    end

    # Validate the model
    st_ = Lux.testmode(train_state.states)
    for (x, y) in data_loaders.val_loader
        ŷ, st_ = model(x, train_state.parameters, st_)
        loss = lossfn(ŷ, y)
        @printf "Validation: Loss %4.5f \n" loss
    end
end



## Predict
inputs = data.q_star[:,:]
x = scale_input(inputs, scaling.in_scaling)
st_ = Lux.testmode(train_state.states)
y_pred = Array{Float32,2}(undef, 6, size(x,2))
for i in 1:size(x,2)
    ip = dev(reshape(x[:,i],(6,1)))
    y, st_ = Lux.apply(train_state.model, ip, train_state.parameters, st_)
    y_pred[:,i] = cpu(y)
end
y_pred = scale_output(Array(y_pred), scaling.out_scaling)
plot_time_series(Array(y_pred), qois, "Predicted dQ", ref = data.dQ)

## Save model
filename = @__DIR__()*"/output/trained_models/ANN_tanh_regularized_hist3.jld2"
save(filename, "model_dict", model_dict, "parameters", tstate.parameters |> cpu, "states", tstate.states|> cpu, "scaling", scaling)