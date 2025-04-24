using JLD2
using DataFrames

fixed_parameters = (tracking_noise = 0.001,
                    hist_len = 10,
                    hist_var = :q_star_q,
                    train_range = (400,4000),
                    lambda = 0.0,
                    indep_normals = false,
                    include_predictor = true,
                    n_replicas = 10,
                    fitted_qois = [1,2,3,4,5,6],
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
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.005)) # succes with noise in linreg targets
i += 1 #6
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = true, tracking_noise = 0.005))
i += 1 #7
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.005, n_replicas = 5))  # no longer noise in linreg targets
i += 1 #8
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.01, n_replicas = 5))
i += 1 #9
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.005, hist_len = 5, n_replicas = 5))
i += 1 #10
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = true, tracking_noise = 0.005, hist_len = 5, n_replicas = 5))
i += 1 #11
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = true, tracking_noise = 0.01, hist_len = 10, n_replicas = 5, lambda = 0.01))
i += 1 #12
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.01, hist_len = 3, n_replicas = 5, lambda = 0.01))
i += 1 #13
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.01, hist_len = 5, n_replicas = 4, lambda = 0.01))
i += 1 #14
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.01, hist_len = 5, n_replicas = 4, lambda = 0.01, fitted_qois = [5,6]))
i += 1 #15
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.01, hist_len = 5, n_replicas = 4, lambda = 0.01, fitted_qois = [3,4,5,6]))
i += 1 #16
push!(inputs, (name = "LinReg$i", fixed_parameters..., indep_normals = false, tracking_noise = 0.005, hist_len = 5, n_replicas = 4, lambda = 0.01, fitted_qois = [3,4,5,6]))
save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)

# read inputs
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
inputs_df = DataFrame(inputs)