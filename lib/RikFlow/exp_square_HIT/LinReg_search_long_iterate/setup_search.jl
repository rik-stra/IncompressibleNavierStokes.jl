using JLD2
using DataFrames

fixed_parameters = (track_file = "/../output/new/data_track2_dns512_les64_Re2000.0_tsim100.0.jld2",
                    hist_len = 10,
                    hist_var = :q_star_q,
                    train_range = (1,4000),
                    include_predictor = true,
                    n_replicas = 10,
                    normalization = :standardise)

varying_parameters = (hist_len = [5, 10, 20], hist_var = [:q, :q_star], include_predictor = [true, false])


i = 0
inputs = []
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters...))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 10))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 10, hist_var = :q))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 10, track_file = "/../output/new/data_track2_dns512_les64_trackingnoise_0.1_Re2000.0_tsim10.0.jld2" ))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 10, track_file = "/../output/new/data_track2_dns512_les64_trackingnoise_0.001_Re2000.0_tsim10.0.jld2" ))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 10, track_file = "/../output/new/data_track2_dns512_les64_trackingnoise_0.001_Re2000.0_tsim10.0.jld2" ))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 10, track_file = "/../output/new/data_track2_dns512_les64_trackingnoise_0.005_Re2000.0_tsim10.0.jld2" ))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 10, track_file = "/../output/new/data_track2_dns512_les64_trackingnoise_mu_0.001_Re2000.0_tsim10.0.jld2" ))
i += 1
push!(inputs, (name = "LinReg$i", fixed_parameters..., hist_len = 10, hist_var = :q, track_file = "/../output/new/data_track2_dns512_les64_trackingnoise_mu_0.001_Re2000.0_tsim10.0.jld2" ))
save(@__DIR__()*"/inputs.jld2", "inputs", inputs)
inputs_df = DataFrame(inputs)

# read inputs
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
inputs_df = DataFrame(inputs)