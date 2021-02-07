# What modules / packages do we depend on
using Random
using LinearAlgebra
using Printf
using Plots

# Import custom types and routines
using MyGEMM.Types: TilingParams
using MyGEMM.TransposeNoPacking: multiply_transpose_no_packing!
using MyGEMM.TransposeMultithreading: multiply_transpose!

Random.seed!(777) # To ensure repeatability
BLAS.set_num_threads(6) # Don't let BLAS use lots of threads (since we are not multi-threaded yet!)

# What precision numbers to use
FT = Float64

# Block matrix-matrix multiply parameters
if FT == Float64
    _mc = 96
    _nc = 2048
    _kc = 256
    _mr = 8
    _nr = 6
    _vl = 4
elseif FT == Float32
    _mc = 2 * 96
    _nc = 2 * 2048
    _kc = 2 * 256
    _mr = 16
    _nr = 12
    _vl = 8
end
params = TilingParams{_mc, _nc, _kc, _mr, _nr, _vl}()

# Different implementations
refgemm!(C, A, B) = mul!(C, A, B', one(eltype(C)), one(eltype(C)))
mygemm_no_packing!(C, A, B) = multiply_transpose_no_packing!(C, A, B, params::TilingParams)
mygemm_packing!(C, A, B) = multiply_transpose!(C, A, B, params::TilingParams)

# Benchmarking routine
function evaluate(gemm!, bench_dims, num_reps)
  num_runs = length(bench_dims)
  performances = []

  # Do loop over varying matrix sizes
  for nmk in bench_dims
    n = m = k = nmk
    gflops = 2 * m * n * k * 1e-09

    # Create some random initial data
    A = rand(FT, m, k)
    B = rand(FT, k, n)
    C = rand(FT, m, n)

    # Make a copy of C for resetting data later
    C_old = copy(C)

    # "truth"
    C_ref = C + A * B'

    # Compute the timings
    best_time = typemax(FT)
    for iter = 1:num_reps
      # Reset C to the original data
      C .= C_old;
      run_time = @elapsed gemm!(C, A, B);
      best_time = min(run_time, best_time)
    end
    
    # Make sure that we have the right answer!
    @assert C ≈ C_ref
    
    # Store data
    best_perf = gflops / best_time
    push!(performances, best_perf)
  end
  
  return performances
end

# Do performance evaluation
num_reps = 3
bench_dims = 48:48:2000
performances = []
for gemm! in [refgemm!, mygemm_no_packing!, mygemm_packing!]
  push!(performances, evaluate(gemm!, bench_dims, num_reps))
end

# Make simple performance plot
p0 = plot(bench_dims, performances[1], linewidth=2, label="reference", legend=:bottom, xlim=(minimum(bench_dims), maximum(bench_dims)))
plot!(p0, bench_dims, performances[2], linewidth=2, label="toby's GEMM no packing (nthreads=1)")
plot!(p0, bench_dims, performances[3], linewidth=2, label="toby's GEMM packing")
xlabel!(p0, "Matrix size")
ylabel!(p0, "GFLOPS")
title!(p0, "Performance evaluation (6 threads)")
savefig("performance_mt.png")
