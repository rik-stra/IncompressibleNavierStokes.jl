using JLD2
using DataFrames

fixed_parameters = (track_file = "/data_track.jld2",
                    hist_len = 5,
                    hist_var = :q,
                    lr = 0.1f0,
                    lambda = 0.1f0,
                    n_replicas = 10,
                    hidden_size = 64,
                    n_layers = 5,
                    batchsize = 64,
                    normalization = :minmax)

varying_parameters = (hist_len = [0, 1, 3, 10, 20],
                      lambda = [0.0f0, 0.01f0, 0.2f0])


i = 1
inputs = [(name = "ANN$i", fixed_parameters...)]
for h in varying_parameters.hist_len
    i += 1
    push!(inputs, (name = "ANN$i", fixed_parameters..., hist_len = h))
end
for l in varying_parameters.lambda
    i += 1
    push!(inputs, (name = "ANN$i", fixed_parameters..., lambda = l))
end

i += 1
push!(inputs, (name = "ANN$i", fixed_parameters..., lambda = 0.02f0, hist_len = 20, lr = 0.01f0, n_replicas = 2))

i += 1
push!(inputs, (name = "ANN$i", fixed_parameters..., lambda = 0.06f0, hist_len = 50, lr = 0.05f0, n_replicas = 2))

i += 1
push!(inputs, (name = "ANN$i", fixed_parameters..., lambda = 0.06f0, hist_len = 50, lr = 0.05f0, n_replicas = 2, hist_var = :q_star))

i += 1
push!(inputs, (name = "ANN$i", fixed_parameters..., lambda = 0.1f0, hist_len = 20, lr = 0.1f0, n_replicas = 10, hist_var = :q_star))

save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)

# read inputs
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
inputs_df = DataFrame(inputs)