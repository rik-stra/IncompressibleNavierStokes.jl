using JLD2
using DataFrames

fixed_parameters = (track_file = "/../ANN_search/data_track.jld2",
                    hist_len = 5,
                    hist_var = :q_star,
                    n_replicas = 10,
                    normalization = :standardise)

varying_parameters = (hist_len = [0, 1, 3, 10, 20, 50],)


i = 1
inputs = [(name = "LinReg$i", fixed_parameters...)]
for h in varying_parameters.hist_len
    i += 1
    push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = h))
end

save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)

# read inputs
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
inputs_df = DataFrame(inputs)