# perfom a HF simulation



using Random
using CairoMakie
using JLD2
using RikFlow
using IncompressibleNavierStokes

dns = 123

# generate a random number
rng_DNS = Xoshiro(dns)
r = rand(rng_DNS, 3)

println("File successfully run! The random number generated is: ", r)
