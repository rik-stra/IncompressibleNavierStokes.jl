using JLD2
using Random

a = rand(1000)
f_name = @__DIR__() * "/output/jld_test_file.jld2"

# Save data to JLD2 file
jldsave(f_name; a)