using JLD2
using DataFrames

fixed_parameters = (tracking_noise = 0,
                    hist_len = 5,
                    lambda = 0.0,
                    model_noise = :MVG,   # :MVG, :tracking_noise ,:no_noise
                    train_range = (100,2000),
                    
                    n_replicas = 5,
                    hist_var = :q_star_q,
                    indep_normals = false,
                    include_predictor = true,
                    fitted_qois = [1,2,3,4,5,6],
                    normalization = :normal,)

i = 0
inputs = []
hist_lens = [5,10]
lambdas = [0.0, 1e-6, 1e-3, 1e-2, 1e-1]

for hist_len in hist_lens
    for lambda in lambdas
        i += 1
        push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = hist_len, lambda = lambda))
    end
end

hist_lens = [5,10]
lambdas = [1e-4, 1e-3]

for hist_len in hist_lens
    for lambda in lambdas
        i += 1
        push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = hist_len, lambda = lambda))
    end
end

save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)



### old inputs
# i += 1
# push!(inputs, (name = "LinReg$i", fixed_parameters...))
# i += 1
# push!(inputs, (name = "LinReg$i", fixed_parameters..., fitted_qois = [1,2,3,4,5,6]))
# i += 1
# push!(inputs, (name = "LinReg$i", fixed_parameters..., fitted_qois = [1,2,3,4,5,6], train_range = (100,2000), n_replicas = 1))
# i += 1
# push!(inputs, (name = "LinReg$i", fixed_parameters..., lambda = 0.01, fitted_qois = [1,2,3,4,5,6], train_range = (100,2000), n_replicas = 1))
# i += 1
# push!(inputs, (name = "LinReg$i", fixed_parameters..., lambda = 0.0, fitted_qois = [1,3,4,5,6], train_range = (100,2000), n_replicas = 1))
# i += 1
# push!(inputs, (name = "LinReg$i", fixed_parameters..., normalization = :normal, lambda = 0.000001, fitted_qois = [1,2,3,4,5,6], train_range = (100,2000), n_replicas = 1))
# save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
# inputs_df = DataFrame(inputs)

