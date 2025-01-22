using JLD2
using DataFrames

fixed_parameters = (tracking_noise = 0.001,
                    hist_len = 10,
                    hist_var = :q_star_q,
                    train_range = (400,4000),
                    indep_normals = false,
                    include_predictor = true,
                    n_replicas = 10,
                    normalization = :standardise)


i = 0
inputs = []
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters...))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters...)) # with spinnup
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = true))

i += 1 #4
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = true))
i += 1 #5
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.005))
i += 1 #6
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = true, tracking_noise = 0.005))
i += 1 #7
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.01))
i += 1 #8
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.0))
i += 1 #9
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.005, hist_len = 5))
i += 1 #10
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = true, tracking_noise = 0.005, hist_len = 5))
save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)

# read inputs
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
inputs_df = DataFrame(inputs)