using JLD2
using CairoMakie
using RikFlow
using Distributions
using LinearAlgebra

data_old = load(@__DIR__()*"/data_track_old.jld2", "data_track");
data_new = load(@__DIR__()*"/data_track.jld2", "data_track");

mean(data_old.dQ, dims=2)
mean(data_new.dQ, dims=2)

mean(data_old.q, dims=2)
mean(data_new.q, dims=2)

mean(data_old.q_star, dims=2)
mean(data_new.q_star, dims=2)


function create_history(hist_len, q_star, q, dQ)
    if hist_len == 0
        return q_star, dQ
    end
    qs = [q[:,hist_len-i+1:end-i+1] for i in 1:hist_len]
    return vcat(q_star[:,hist_len:end], qs...), dQ[:,hist_len:end]
end


function fit_LinReg(; hist_var, hist_len, data)

    if hist_var == :q
        inputs,outputs = create_history(hist_len, data.q_star[:,1:3000], data.q[:,1:3000], data.dQ[:,1:3000])
    elseif hist_var == :q_star
        inputs,outputs = create_history(hist_len, data.q_star[:,2:3000], data.q_star[:,1:3000-1], data.dQ[:,2:3000])
    end
    inputs_scaled, in_scaling = RikFlow._normalise(inputs, normalization = :standardise)
    outputs_scaled, out_scaling = RikFlow._normalise(outputs, normalization = :standardise)
    scaling = (;in_scaling, out_scaling)

    inp = cat(inputs_scaled',ones(eltype(inputs_scaled), (size(inputs_scaled,2),1)),dims=2) # add a bias term
    c = inp \ outputs_scaled' 
    #For rectangular A the result is the minimum-norm least squares solution computed by a pivoted QR factorization of A and a rank estimate of A based on the R factor
    preds = inp * c
    stoch_part = outputs_scaled - preds'
    
    # fit MVG
    stoch_distr = fit(MvNormal, stoch_part .|> Float64)

    model = (c= c', stoch_distr= stoch_distr, scaling= scaling, hist_var= hist_var, hist_len= hist_len)
    return model
end

hist_var = :q
hist_len = 6

model_old = fit_LinReg(hist_var=hist_var, hist_len=hist_len, data=data_old)
model_new = fit_LinReg(hist_var=hist_var, hist_len=hist_len, data=data_new)

model_old.c[2,:]
model_old.c[2,6:6:end]
model_new.c[2,6:6:end]

model_old.stoch_distr.μ
model_new.stoch_distr.μ

(model_old.stoch_distr.Σ - model_new.stoch_distr.Σ)./model_old.stoch_distr.Σ