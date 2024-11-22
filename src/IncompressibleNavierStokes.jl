"""
Energy-conserving solvers for the incompressible Navier-Stokes equations.

## Exports

The following symbols are exported by IncompressibleNavierStokes:

$(EXPORTS)
"""
module IncompressibleNavierStokes

using Adapt
using ChainRulesCore
using DocStringExtensions
using FFTW
using EnzymeCore
using EnzymeCore.EnzymeRules
using IterativeSolvers
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll
using LinearAlgebra
using NNlib
using Observables
using PrecompileTools
using Printf
using Random
using SparseArrays
using StaticArrays
using Statistics
using WriteVTK: CollectionFile, paraview_collection, vtk_grid, vtk_save

# Docstring templates
@template MODULES = """
                   $(DOCSTRING)

                   ## Exports

                   $(EXPORTS)
                   """
@template (FUNCTIONS, METHODS) = """
                                 $TYPEDSIGNATURES

                                 $DOCSTRING
                                 """
@template TYPES = """
                  $TYPEDEF

                  $DOCSTRING

                  ## Fields

                  $FIELDS
                  """

"$LICENSE"
license = "MIT"

# # Easily retrieve value from Val
# (::Val{x})() where {x} = x

# General stuff
include("boundary_conditions.jl")
include("grid.jl")
include("setup.jl")
include("pressure.jl")
include("operators.jl")
include("eddyviscosity.jl")
include("matrices.jl")
include("initializers.jl")
include("processors.jl")
include("sciml.jl")
include("solver.jl")
include("utils.jl")

# Time steppers
include("time_steppers/methods.jl")
include("time_steppers/time_stepper_caches.jl")
include("time_steppers/step.jl")
include("time_steppers/RKMethods.jl")

# Precompile workflow
include("precompile.jl")

# Boundary conditions
export PeriodicBC, DirichletBC, SymmetricBC, PressureBC

# Processors
export processor,
    timelogger,
    vtk_writer,
    observefield,
    observespectrum,
    fieldsaver,
    realtimeplotter,
    animator
export fieldplot, energy_history_plot, energy_spectrum_plot

# Setup
export Setup, temperature_equation, scalarfield, vectorfield
export CPU # TODO: This annoys Documenter, need to include this docstring somehow

# 1D grids
export stretched_grid, cosine_grid, tanh_grid

# Pressure solvers
export default_psolver, psolver_direct, psolver_cg, psolver_cg_matrix, psolver_spectral

# Solvers
export solve_unsteady, timestep, create_stepper

# Field generation
export velocityfield, temperaturefield, random_field

# Utils
export getoffset, splitseed, plotgrid, save_vtk, get_lims

# ODE methods
export AdamsBashforthCrankNicolsonMethod, OneLegMethod, RKMethods, LMWray3

# Operators
export apply_bc_u,
    apply_bc_p,
    apply_bc_temp,
    applybodyforce,
    applypressure,
    convection_diffusion_temp,
    convection,
    diffusion,
    dissipation,
    dissipation_from_strain,
    divergence,
    eig2field,
    get_scale_numbers,
    gravity,
    kinetic_energy,
    interpolate_u_p,
    interpolate_ω_p,
    laplacian,
    laplacian_mat,
    momentum,
    poisson,
    pressure,
    pressuregradient,
    project,
    scalewithvolume,
    smagorinsky_closure,
    tensorbasis,
    total_kinetic_energy,
    vorticity,
    Dfield,
    Qfield

# SciML operations
export create_right_hand_side, right_hand_side!

end
