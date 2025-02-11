using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics
using RikFlow

fig_folder = @__DIR__()*"/output/figures_paper"
if !isdir(fig_folder)
    mkdir(fig_folder)
end

filename =  @__DIR__()*"/output/u_start_spinnup_512_Re2000.0_freeze_10_tsim4.0.jld2"
u_start = stack(load(filename, "u_start"));

n = 512
Δx = 1/n
axis_x = range(0.0, 1., n + 1)
setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
state = (;u = u_start, t=0., temp=0);
        
# save to vtk
#save_vtk(state; setup, filename = @__DIR__()*"/output/vtks", fieldnames = (:velocity, :Qfield))
scales = get_scale_numbers(u_start, setup)
fig = energy_spectrum_plot(state; setup, npoint = 100, sloperange = [2,16], v_lines = [scales.λ, scales.η, Δx], slopeoffset = 1.8, scale_numbers = scales, plot_wavelength = true)
display(fig)
v = [scales.λ, scales.η, 1/n]
v_labels = ["λ", "η", "Δx"]
for i in 1:3
    text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1e-12*1.2), align = (:left, :bottom), color = :black)
end
display(fig)
save(fig_folder*"/energy_spectrum_afterspinup_512_Re2000.0_freeze_10_tsim4.png", fig)

## energy spectrum coarse grained
filename = @__DIR__()*"/output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
ref_data = load(filename, "data_train");

u_start_lf = ref_data.data[1].u[1];
heatmap(u_start_lf[end-1, :, :, 1]) # initial coarse field
heatmap(u_start[end-1, :, :, 1])
n = 64
Δx = 1/n
axis_x = range(0.0, 1., n + 1)
setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
state = (;u = u_start_lf, t=0., temp=0);
scales_LF = get_scale_numbers(u_start_lf, setup)
fig = energy_spectrum_plot(state; setup, npoint = 100, sloperange = [2,16], v_lines = [6.5,15.5,32], slopeoffset = 1.8, scale_numbers = scales)
v_labels = ["[0,6]", "[7,15]", "[16,32]"]
v = [3, 11, 24]
for i in 1:3
    text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1*0.5), align = (:center, :center), color = :black)
end
display(fig)
save(fig_folder*"/energy_spectrum_afterspinup_coarse_grained_Re2000.0_freeze_10_tsim4.png", fig)