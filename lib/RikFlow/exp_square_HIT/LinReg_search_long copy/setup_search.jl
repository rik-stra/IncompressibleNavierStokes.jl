using JLD2
using DataFrames

fixed_parameters = (track_file = "/../output/new/data_track2_dns512_les64_Re2000.0_tsim100.0.jld2",
                    hist_len = 5,
                    hist_var = :q_star,
                    train_range = (1,4000),
                    include_predictor = true,
                    n_replicas = 10,
                    normalization = :standardise)

varying_parameters = (hist_len = [5, 10, 20], hist_var = [:q, :q_star], include_predictor = [true, false])


i = 0
inputs = []
for pred in varying_parameters.include_predictor
    for hv in varying_parameters.hist_var
        for h in varying_parameters.hist_len
            i += 1
            push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = h, hist_var = hv, include_predictor = pred))
        end
    end
end
hist_len = [5, 10, 20, 3]
for h in hist_len
    i += 1
    push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = h, hist_var = :q_star_q, include_predictor = true))
end
hist_len = [5, 10]
for h in hist_len
    i += 1
    push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = h, hist_var = :q_star_q, include_predictor = true, train_range = (1,40000), n_replicas = 2))
end
hist_len = [5, 10]
for h in hist_len
    i += 1
    push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = h, hist_var = :q_star_q, include_predictor = true, train_range = (400,4000), n_replicas = 10))
end
hist_len = [5, 10]
for h in hist_len
    i += 1
    push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = h, hist_var = :q_star_q, include_predictor = true, train_range = (400,40000), n_replicas = 10))
end
#push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 100, hist_var = :q))

save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)

# read inputs
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
inputs_df = DataFrame(inputs)