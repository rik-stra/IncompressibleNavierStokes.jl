using JLD2
using DataFrames
using IncompressibleNavierStokes
using RikFlow

index_range = [2,2]
linreg_params_table = DataFrame(load(@__DIR__()*"output/TO_LRS/inputs_example.jld2", "inputs"))[index_range[1]:index_range[2],:]
# load reference data
hf_file = @__DIR__()*"/output/paper_data_channel/HF/HF_channel_512_512_256_to_64_64_32_tsim15.0.jld2"
q_ref = stack(load(hf_file)["f"].data[1].qoi_hist[1:5:10001]);
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]


ks_dists_replicas = []
ks_dist_ensemble = []
n_unstable = []
for (name, n_replicas) in zip(linreg_params_table.name, linreg_params_table.n_replicas)
    stable_sims = []
    unstable = 0
    for i in 1:n_replicas
        file = @__DIR__()*"/output/TO_LRS/$(name)/LF_online_channel_to_64_64_32_tsim10.0_repl_$(i).jld2"
        if isfile(file)
            stable_sims = push!(stable_sims, i)
        else
            println("Simulation $(name) replica $(i) was not stable")
            unstable += 1
        end
    end
    
    q_rep = [load(@__DIR__()*"/output/TO_LRS/$(name)/LF_online_channel_to_64_64_32_tsim10.0_repl_$(i).jld2", "data").q for i in stable_sims]    
    qs = cat(q_rep..., dims = 2)
   
    # check if sim was stable for full lenght
    
    for i = 1:length(stable_sims)
        if length(q_rep[i][1,:]) < 2000
            println("Simulation $(name) replica $(i) was not stable")
            unstable += 1
        end
    end

    if size(q_rep,1) > 0
        temp = []
        for r in 1:size(q_rep,1)
            ks = [ks_dist(q_ref[i,:], q_rep[r][i,:])[1] for i in 1:length(qois)]
            ks = [sum(ks), ks...]
            push!(temp, ks)
        end
        ks_ensemble = [ks_dist(q_ref[i,:], qs[i,:])[1] for i in 1:length(qois)]
        ks_ensemble = [sum(ks_ensemble), ks_ensemble...]
    else
        temp = nothing
        ks_ensemble = nothing
    end
    
    push!(ks_dists_replicas, temp)
    push!(ks_dist_ensemble, ks_ensemble)
    push!(n_unstable, unstable)
end

linreg_params_table.ks_dist_replicas = ks_dists_replicas
linreg_params_table.ks_dist_ensemble = ks_dist_ensemble
linreg_params_table.n_unstable = n_unstable

ispath(@__DIR__()*"/output/ks_data") || mkdir(@__DIR__()*"/output/ks_data")
save(@__DIR__()*"/output/ks_data/ks_dists_range_$(index_range[1])_$(index_range[2]).jld2", "ks_table", linreg_params_table)

