using JLD2
using DataFrames

fixed_parameters = (tracking_noise = 0,
                    hist_len = 5,
                    lambda = 0.0,
                    model_noise = :MVG,   # :MVG, :tracking_noise ,:no_noise
                    train_range = (100,1000),
                    
                    n_replicas = 5,
                    hist_var = :q_star_q,
                    indep_normals = false,
                    include_predictor = true,
                    fitted_qois = [1,2,3,4],
                    normalization = :standardise,)

i = 0
inputs = []
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters...))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., fitted_qois = [1,2,3,4,5,6]))
                
save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)

