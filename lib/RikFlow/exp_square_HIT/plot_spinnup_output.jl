using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics
using RikFlow


###############################################
# Energy spectra initial turbulent field (after spinnup)
###############################################

figs_folder = @__DIR__()
#filename =  @__DIR__()*"/output/paper_data_HIT/u_start_spinnup_512_Re2000.0_freeze_10_tsim4.0.jld2"
filename = @__DIR__()*"/output_spinnup/u_start_spinnup_800_Re2000.0_freeze_10_tsim4.0.jld2"
u_start_800 = stack(load(filename, "u_start"));
n = 800
Δx_hf = 1/n
axis_x = range(0.0, 1., n + 1)
setup_HF = rf_setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
state_HF = (;u = u_start_800, t=0., temp=0);

#filename =  @__DIR__()*"/output/u_start_spinnup_128_Re2000.0_freeze_10_tsim0.2.jld2"
filename = @__DIR__()*"/output_spinnup/u_start_spinnup_512_Re2000.0_freeze_10_tsim4.0.jld2"
u_start_512 = stack(load(filename, "u_start"));
n = 512
Δx = 1/n
axis_x = range(0.0, 1., n + 1)
setup_LF = rf_setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
state_LF = (;u = u_start_512, t=0., temp=0);

scales = turbulence_statistics(u_start_800, setup_HF, 1/setup_HF.Re)
println("Scale numbers: $(scales)")


fig = rf_energy_spectrum_plot([state_HF, state_LF]; setup = [setup_HF, setup_LF], npoint = 100, sloperange = [2,16], v_lines = [scales.l_tay, scales.l_kol, Δx_hf], slopeoffset = 1.9,
 scale_numbers = scales, plot_wavelength = true, plot_n_spectra = 2)
#display(fig)
v = [scales.l_tay, scales.l_kol, Δx_hf]
v_labels = ["λ", "η", "Δx"]
for i in 1:3
    text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1e-15*1.2), align = (:left, :bottom), color = :black)
end
display(fig)
save(figs_folder*"/energy_spectrum_afterspinup_refined.pdf", fig)
