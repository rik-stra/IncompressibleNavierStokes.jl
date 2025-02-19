using JLD2
using DataFrames
using IncompressibleNavierStokes
using RikFlow

        
linreg_params_table = DataFrame(load(@__DIR__()*"/inputs.jld2", "inputs"))
# load reference data
filename = @__DIR__()*"/../output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
ref_data = load(filename, "data_train");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
q_ref = stack(ref_data.data[1].qoi_hist)

ks_dists_replicas = []
ks_dist_ensemble = []
for (name, n_replicas) in zip(linreg_params_table.name, linreg_params_table.n_replicas)

    q_rep = [load(@__DIR__()*"/output/online/$(name)/data_online_dns512_les64_Re2000.0_tsim100.0_replica$(i)_rand_initial_dQ.jld2", "data_online").q for i in 1:n_replicas]    
    qs = cat(q_rep..., dims = 2)
    
    temp = []
    for r in 1:n_replicas
        ks = [ks_dist(q_ref[i,:], q_rep[r][i,:])[1] for i in 1:length(qois)]
        ks = [sum(ks), ks...]
        push!(temp, ks)
    end
    
    ks_ensemble = [ks_dist(q_ref[i,:], qs[i,:])[1] for i in 1:length(qois)]
    ks_ensemble = [sum(ks_ensemble), ks_ensemble...]

    push!(ks_dists_replicas, temp)
    push!(ks_dist_ensemble, ks_ensemble)
end

linreg_params_table.ks_dist_replicas = ks_dists_replicas
linreg_params_table.ks_dist_ensemble = ks_dist_ensemble

save(@__DIR__()*"/output/ks_dists.jld2", "ks_table", linreg_params_table)
#linreg_params_table[!,"ks_dist_ensemble"]

