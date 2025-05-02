using JLD2
using DataFrames

fixed_parameters = (tracking_noise = 0.1,
                    hist_len = 5,
                    lambda = 0.1,
                    model_noise = :MVG,   # :MVG, :tracking_noise ,:no_noise
                    train_range = (400,4000),
                    
                    n_replicas = 5,
                    hist_var = :q_star_q,
                    indep_normals = false,
                    include_predictor = true,
                    fitted_qois = [1,2,3,4,5,6],
                    normalization = :standardise,
                    )

noise_levels = [0, 0.1, 0.01]
hist_lens = [5, 20]
labs = [0, 0.1, 0.01]
model_noises = [:MVG, :tracking_noise, :no_noise]

i = 0
inputs = []
for hist_len in hist_lens
    for tracking_noise in noise_levels
        for lambda in labs
            for model_noise in model_noises
                if !(model_noise == :tracking_noise && tracking_noise == 0)
                    i += 1
                    push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len, tracking_noise, lambda, model_noise))
                end
            end
        end
    end
end
hist_lens = [0,1,3,50,80]
lambdas = [0.01, 0]
for hist_len in hist_lens
    for lambda in lambdas
        
        tracking_noise = 0.0
        model_noise = :MVG
                    
        i += 1
        push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len, tracking_noise, lambda, model_noise))
    end
end
hist_lens = [4,8,10,25,30]
lambdas = [0.01, 0]
for hist_len in hist_lens
    for lambda in lambdas
        
        tracking_noise = 0.0
        model_noise = :MVG
                    
        i += 1
        push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len, tracking_noise, lambda, model_noise))
    end
end
hist_lens = [6,7,40,60]
lambdas = [0.01, 0]
for hist_len in hist_lens
    for lambda in lambdas
        
        tracking_noise = 0.0
        model_noise = :MVG
                    
        i += 1
        push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len, tracking_noise, lambda, model_noise))
    end
end

save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)

#####
# small training data
#####

train_ranges = [(400, 2000), (400, 1000)]
noise_levels = [0.0, 0.1, 0.01]
hist_lens = [5, 10, 20, 30]
labs = [0, 0.1, 0.01]
print("Start small training data $(i+1)")
for hist_len in hist_lens
    for tracking_noise in noise_levels
        for lambda in labs
            for train_range in train_ranges
                    i += 1
                    push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len, tracking_noise, lambda, train_range))
            end
        end
    end
end
print("End small training data $(i)")
train_range = (400, 3000)
tracking_noise = 0.0
hist_lens = [5, 10]
labs = [0, 0.1, 0.01]
print("Start extra training data $(i+1)")
for hist_len in hist_lens
        for lambda in labs
            i += 1
            push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len, tracking_noise, lambda, train_range)) 
        end
end
train_range = (400, 4000)
lambda = 0.1
hist_len = 10
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len, tracking_noise, lambda, train_range)) 
print("End extra training data $(i)")
save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)


# read inputs
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
inputs_df = DataFrame(inputs)

# find linreg h = 10
inputs_df[inputs_df.hist_len .== 10 .&& inputs_df.tracking_noise .== 0.0, :]