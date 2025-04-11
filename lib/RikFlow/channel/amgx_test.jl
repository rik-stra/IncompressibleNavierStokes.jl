using LinearAlgebra
using SparseArrays
using CUDA
using AMGX



AMGX.initialize()
config = AMGX.Config(Dict(
         "config_version" => "2",
         "solver" => "PCGF",
         "tolerance" => 0.0001,
         #"cg:max_iters" => 100,
         "preconditioner(amg_solver)" => "AMG",
         "amg_solver:algorithm" => "CLASSICAL",
         "amg_solver:max_iters" => 1,
         #"amg_solver:cycle" => 'W',
         
    ))
resources = AMGX.Resources(config)

str = ""
store_to_str(amgx_printed::String) = (global str = amgx_printed; nothing)
AMGX.register_print_callback(store_to_str)
c_config = AMGX.Config("")
print(str)
print_stdout(amgx_printed::String) = print(stdout, amgx_printed)
AMGX.register_print_callback(print_stdout)


function poisson_3d_sparse(n)
    N = n^3
    e = ones(n)

    # 1D Laplacian matrix
    T = spdiagm(-1 => -e[1:end-1], 0 => 2e, 1 => -e[1:end-1])
    I = spdiagm(0 => ones(n))   # Sparse identity matrix

    # 3D Laplacian via Kronecker sums
    L = kron(I, kron(I, T)) + kron(I, kron(T, I)) + kron(T, kron(I, I))

    return L
end


A = Float32.(poisson_3d_sparse(200)); 
A = CUDA.CUSPARSE.CuSparseMatrixCSR{}(A)

b = rand(Float32,200^3);
b = CuArray(b)

v = AMGX.AMGXVector(resources, AMGX.dFFI)
AMGX.upload!(v, b)

matrix = AMGX.AMGXMatrix(resources, AMGX.dFFI)
AMGX.upload!(matrix, 
    ((A.rowPtr).-Int32(1)), # row_ptrs
    ((A.colVal).-Int32(1)), # col_indices
    (A.nzVal) # data
)



solver = AMGX.Solver(resources, AMGX.dFFI, config)


@time AMGX.setup!(solver, matrix)

x = AMGX.AMGXVector(resources, AMGX.dFFI)
AMGX.set_zero!(x,size(b,1))
@time AMGX.solve!(x, solver, v)
Vector(x)

close(x)
close(v)
close(matrix)
close(solver)
close(resources)
close(config)
AMGX.finalize()