using JLD2
using DataFrames

fixed_parameters = (track_file = "/../LinReg_search_old/data_track.jld2",
                    hist_len = 5,
                    hist_var = :q_star,
                    n_replicas = 10,
                    normalization = :standardise)

varying_parameters = (hist_len = [0, 1, 3, 5, 10, 20, 50], hist_var = [:q, :q_star])


i = 0
inputs = []
for hv in varying_parameters.hist_var
    for h in varying_parameters.hist_len
        i += 1
        push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = h, hist_var = hv))
    end
end
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 100, hist_var = :q))

save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)

# read inputs
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
inputs_df = DataFrame(inputs)