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
filename = @__DIR__()*"/output_spinnup/u_start_spinnup_900_Re2000.0_freeze_10_tsim4.0.jld2"
u_start = stack(load(filename, "u_start"));
n = 900
Δx = 1/n
axis_x = range(0.0, 1., n + 1)
setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
state = (;u = u_start, t=0., temp=0);
        
scales = get_scale_numbers(u_start, setup)
println("Scale numbers: $(scales)")
fig = energy_spectrum_plot(state; setup, npoint = 100, sloperange = [2,16], v_lines = [scales.λ, scales.η, Δx], slopeoffset = 1.8, scale_numbers = scales, plot_wavelength = true)
#display(fig)
v = [scales.λ, scales.η, 1/n]
v_labels = ["λ", "η", "Δx"]
for i in 1:3
    text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1e-12*1.2), align = (:left, :bottom), color = :black)
end
display(fig)
save(figs_folder*"/energy_spectrum_afterspinup.pdf", fig)
