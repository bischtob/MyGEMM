using LinearAlgebra
using Test
using Random
using MyGEMM.TransposeNoPacking: multiply_transpose_no_packing!
using MyGEMM.TransposeMultithreading: multiply_transpose!
using MyGEMM.Types: TilingParams

Random.seed!(777)

params = TilingParams{96, 2048, 256, 8, 6, 4}()

@testset "TransposeNoPacking" begin
  # Square test
  n, m, k = 48, 48, 48
  A, B, C = rand(m, k), rand(n, k), rand(m, n)
  C0 = C + A * B'
  multiply_transpose_no_packing!(C, A, B, params)
  @test C0 ≈ C
  
  # Rectangle test
  n, m, k = 48, 2*48, 3*48
  A, B, C = rand(m, k), rand(n, k), rand(m, n)
  C0 = C + A * B'
  multiply_transpose_no_packing!(C, A, B, params)
  @test C0 ≈ C
end

@testset "TransposeMultithreading" begin
  # Square test
  n, m, k = 48, 48, 48
  A, B, C = rand(m, k), rand(n, k), rand(m, n)
  C0 = C + A * B'
  multiply_transpose!(C, A, B, params)
  @test C0 ≈ C
  
  # Rectangle test
  n, m, k = 48, 2*48, 3*48
  A, B, C = rand(m, k), rand(n, k), rand(m, n)
  C0 = C + A * B'
  multiply_transpose!(C, A, B, params)
  @test C0 ≈ C
end
