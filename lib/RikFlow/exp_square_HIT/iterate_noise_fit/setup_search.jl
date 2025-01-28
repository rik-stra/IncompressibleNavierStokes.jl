using JLD2
using DataFrames

fixed_parameters = (tracking_noise = "var",
                    hist_len = 5,
                    hist_var = :q_star_q,
                    train_range = (400,4000),
                    lambda = 0.1,
                    indep_normals = false,
                    include_predictor = true,
                    n_replicas = 4,
                    fitted_qois = [1,2,3,4,5,6],
                    normalization = :standardise)


i = 0
inputs = []
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters...))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 20))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 10))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 5))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 3))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 200))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 50))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 100))

save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)

# read inputs
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
inputs_df = DataFrame(inputs)