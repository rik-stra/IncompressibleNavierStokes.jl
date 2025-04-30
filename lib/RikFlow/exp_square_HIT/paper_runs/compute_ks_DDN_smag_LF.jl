using JLD2
using DataFrames
using IncompressibleNavierStokes
using RikFlow

names = ["DDN", "smag", "nomodel"]
file_names = ["/../output/new/MVG/data_online_dns512_les64_Re2000.0_tsim100.0_replica",
              "/../output/new/smag/data_smag_0.071_dns512_les64_Re2000.0_tsim100.0.jld2",
              "/../output/new/data_no_sgs_dns512_les64_Re2000.0_tsim100.0.jld2"]
n_replicas_list = [5, 1 ,1]

# load reference data
filename = @__DIR__()*"/../output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
ref_data = load(filename, "data_train");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
q_ref = stack(ref_data.data[1].qoi_hist)

ks_dists_replicas = []
ks_dist_ensemble = []
n_unstable = []
for (name, n_replicas, file_str) in zip(names, n_replicas_list, file_names)
    stable_sims = []
    unstable = 0
    if n_replicas>1
        for i in 1:n_replicas
            file = @__DIR__()*file_str*"$(i).jld2"
            if isfile(file)
                stable_sims = push!(stable_sims, i)
            else
                println("Simulation $(name) replica $(i) was not stable")
                unstable += 1
            end
        end
        
        q_rep = [load(@__DIR__()*file_str*"$(i).jld2", "data_online").q for i in stable_sims]
    else
        q_rep = [load(@__DIR__()*file_str, "data_online").q]
    end
    qs = cat(q_rep..., dims = 2)
    
    # check if sim was stable for full lenght
    
    for i = 1:size(q_rep,1)
        if length(q_rep[i][1,:]) < 40000
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

linreg_params_table = DataFrame(name = names, n_replicas = n_replicas_list)
linreg_params_table.ks_dist_replicas = ks_dists_replicas
linreg_params_table.ks_dist_ensemble = ks_dist_ensemble
linreg_params_table.n_unstable = n_unstable
linreg_params_table.tracking_noise = [0.0,0.0,0.0]
linreg_params_table.model_noise = [:MVG,:MVG,:MVG]
linreg_params_table.hist_len = [-5,-10,-15]

save(@__DIR__()*"/output/ks_dists_DDN_smag_lf.jld2", "ks_table", linreg_params_table)
#linreg_params_table[!,"ks_dist_ensemble"]

