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
using IterativeSolvers
using KernelAbstractions
using LinearAlgebra
using Makie
using NNlib
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
include("matrices.jl")

# Time steppers
include("time_steppers/methods.jl")
include("time_steppers/time_stepper_caches.jl")
include("time_steppers/step.jl")
include("time_steppers/isexplicit.jl")
include("time_steppers/lambda_max.jl")
include("time_steppers/RKMethods.jl")

# Preprocess
include("create_initial_conditions.jl")

# Processors
include("processors.jl")

# Solvers
# include("solvers/get_timestep.jl")
include("solvers/cfl.jl")
include("solvers/solve_unsteady.jl")

# Utils
include("utils/plotgrid.jl")
include("utils/get_lims.jl")
include("utils/plotmat.jl")
include("utils/spectral_stuff.jl")

# Boundary conditions
export PeriodicBC, DirichletBC, SymmetricBC, PressureBC

# Processors
export processor, timelogger, vtk_writer, fieldsaver, realtimeplotter
export fieldplot, energy_history_plot, energy_spectrum_plot
export animator

# Setup
export Setup, temperature_equation

# 1D grids
export stretched_grid, cosine_grid, tanh_grid

# Pressure solvers
export default_psolver,
    psolver_direct,
    psolver_cg,
    psolver_cg_matrix,
    psolver_spectral,
    psolver_spectral_lowmemory

# Solvers
export solve_unsteady

# Field generation
export create_initial_conditions, random_field

# Utils
export plotgrid, save_vtk
export plotmat

# ODE methods
export AdamsBashforthCrankNicolsonMethod, OneLegMethod, RKMethods

end
