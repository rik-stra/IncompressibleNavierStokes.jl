# This script creates a data set with hyperparameters for the TO LRS model. This allows for systematic hyperparameter search.
using JLD2
using DataFrames

# standard settings
fixed_parameters = (
                    # stuff you want to explore
                    hist_len = 5, # number of history point included in the linear regression
                    lambda = 0,   # regularization strength in linear regression
                    train_range = (1,401), # range of training data to use when fitting linear regression (we used Δt = 0.05, so this is timeunit 0 to 20)
                    n_replicas = 5, # number of replicas to run when evaluating the model online
                    
                    # stuff you might want to explore
                    model_noise = :MVG,   # noise model for residual of linear regression options :MVG multi variate gaussian, :no_noise no noise added to linreg, :model_noise use the same noise as during tracking (see "tracking_noise")
                    fitted_qois = [1,2,3,4,5,6],  # choose which qois to fit the linear regression to.
                    normalization = :standardise, # how to normalize the qois before fitting the linear regression, options: :standardise, :normal, :minmax

                    # stuff you probably don't want to explore 
                    hist_var = :q_star_q,  # include both q_star and q in the history, options: :q, :q_star_q
                    indep_normals = false, # if true: fit MVG with diagonal covariance matrix (so "independent normal distributions")
                    include_predictor = true, # include the predictor q_star in the model inputs
                    tracking_noise = 0,    # add noise to the reference trajectories data (results in a data-assimilation-like problem). If you want to explore this you also need to run multiple tracking simulations with different noise levels and possibly different randomseeds 
                    )


i = 0
inputs = []
hist_lens = [5,10,0,1]
lambdas = [0.0, 0.005, 0.02]

for hist_len in hist_lens
    for lambda in lambdas
        i += 1
        push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = hist_len, lambda = lambda))
    end
end

i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 5, lambda = 0.0, model_noise = :no_noise, n_replicas = 1)) # no noise, single replica
outdir = @__DIR__()*"/output/TO_LRS"
ispath(outdir) || mkpath(outdir)
save(outdir*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)

