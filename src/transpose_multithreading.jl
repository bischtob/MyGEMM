module TransposeMultithreading

using KernelAbstractions.Extras: @unroll
using SIMD: Vec, vload, vstore
using StaticArrays: MArray, MVector
using MyGEMM.Types: TilingParams
using MyGEMM.Utils: packA!, packB!

export multiply_transpose!

"""
  multiply_transpose!(C, A, B, params)

Performs `C += A * B'` with packing and multithreading option. 
"""
@inline function multiply_transpose!(C, A, B, params::TilingParams)
  FT  = eltype(C)
  
  # Get the matrix params
  m, n = size(C)
  k = size(B, 2)

  # Check the matrix params
  @assert size(C) == (m, n)
  @assert size(A) == (m, k)
  @assert size(B) == (n, k)
  
  # Get tiling parameters
  mr = params.mr
  nr = params.nr
  vl = params.vl
  mc = params.mc
  nc = params.nc
  kc = params.kc

  # Since `B_packed` lives in the L3 cache we need to rescale it
  nc = cld(cld(min(n, nc), Threads.nthreads()), nr) * nr
  params = TilingParams{mc, nc, kc, mr, nr, vl}()

  # We assume that the matrices are even divisible by `nr` and `mr`, 
  # that the blocking perfectly partitions the matrix,
  # and, vectors are perfectly divided by the chosen vector unit
  @assert mod(m, mr)  == 0
  @assert mod(n, nr)  == 0
  @assert mod(mc, mr) == 0
  @assert mod(nc, nr) == 0
  @assert mod(mr, vl) == 0

  # Prepare tuples for packing
  A_packed = ntuple(j -> Array{FT}(undef, mc * kc), Threads.nthreads())
  B_packed = ntuple(j -> Array{FT}(undef, kc * nc), Threads.nthreads())

  # Execute C = C + A * B'
  loop5!(C, A, B, A_packed, B_packed, params)
end

@inline function loop5!(C, A, B, A_packed, B_packed, params::TilingParams)
  n = size(C, 2)
  nc = params.nc
  @inbounds Threads.@threads for j1 = 1:nc:n
    j2 = min(n, j1 + nc - 1)
    C_j = @view C[:, j1:j2]
    B_j = @view B[j1:j2, :]
    loop4!(
      C_j, 
      A, 
      B_j,
      A_packed[Threads.threadid()],
      B_packed[Threads.threadid()],
      params
    )
  end
end

@inline function loop4!(C, A, B, A_packed, B_packed, params::TilingParams)
  k = size(A, 2)
  kc = params.kc
  mr = params.mr
  nr = params.nr
  @inbounds for p1 = 1:kc:k
    p2 = min(k, p1 + kc - 1)
    A_p = @view A[:, p1:p2]
    B_p = @view B[:, p1:p2]
    real_kc = (p2 - p1 + 1)
    kc_nr = real_kc * nr
    kc_mr = real_kc * mr
    packB!(B_packed, B_p, params)
    loop3!(C, A_p, A_packed, B_packed, kc_nr, kc_mr, params)
  end
end

@inline function loop3!(C, A, A_packed, B_packed, kc_nr, kc_mr, params::TilingParams)
  m = size(C, 1)
  mc = params.mc
  @inbounds for i1 = 1:mc:m
    i2 = min(m, i1 + mc - 1)
    C_i = @view C[i1:i2, :]
    A_i = @view A[i1:i2, :]
    packA!(A_packed, A_i, params)
    loop2!(C_i, A_packed, B_packed, kc_nr, kc_mr, params)
  end
end

@inline function loop2!(C, A_packed, B_packed, kc_nr, kc_mr, params::TilingParams)
  n = size(C, 2)
  nr = params.nr
  @inbounds for (t, j1) = enumerate(1:nr:n)
    j2 = min(n, j1 + nr - 1)
    C_j = @view C[:, j1:j2]
    tile_rng = (1 + kc_nr * (t - 1)) : (t * kc_nr)
    B_packed_j = @view B_packed[tile_rng]
    loop1!(C_j, A_packed, B_packed_j, kc_mr, params)
  end
end

@inline function loop1!(C, A_packed, B_packed, kc_mr, params::TilingParams)
  m = size(C, 1)
  mr = params.mr
  @inbounds for (t, i1) = enumerate(1:mr:m)
    i2 = min(m, i1 + mr - 1)
    C_i = @view C[i1:i2, :]
    tile_rng = (1 + kc_mr * (t - 1)) : (t * kc_mr)
    A_packed_i = @view A_packed[tile_rng]
    microknl!(C_i, A_packed_i, B_packed, params)
  end
end

@inline function microknl!(C, A, B, params::TilingParams)
  # Get relevant tile params
  m = size(parent(C), 1)
  mr = params.mr
  nr = params.nr
  vl = params.vl
  mr_vl = div(mr, vl)
  pend = div(size(B, 1), nr)
 
  # Set relevant types
  FT = eltype(C)
  VT = Vec{vl, FT}

  # Initialize vector storage
  C_tile = MArray{Tuple{mr_vl, nr}, VT}(undef)
  A_panel = MVector{mr_vl, VT}(undef)

  # Load the columns of the microtile of C
  @inbounds @unroll for j = 1:nr
    @unroll for i = 1:mr_vl
      offset = ((i - 1) * vl + (j - 1) * m) * sizeof(FT)
      C_tile[i, j] = vload(VT, pointer(C) + offset, nothing, Val(true))
    end
  end

  # Do the rank-one updates
  @inbounds @unroll for p = 1:pend
    # Load pieces of `A_panel`
    @unroll for i = 1:mr_vl
      offset = ((p - 1) * mr + (i - 1) * vl) * sizeof(FT)
      A_panel[i] = vload(VT, pointer(A) + offset, nothing, Val(true))
    end
  
    # Do outer product updates
    @unroll for j = 1:nr
      β = B[j + nr * (p - 1)]
      @unroll for i = 1:mr_vl
        C_tile[i, j] = muladd(A_panel[i], β, C_tile[i, j])
      end
    end
  end

  # Write back from vector storage
  @inbounds @unroll for j = 1:nr
    @unroll for i = 1:mr_vl
      offset = ((i - 1) * vl + (j - 1) * m) * sizeof(FT)
      vstore(C_tile[i, j], pointer(C) + offset, nothing, Val(true), Val(true))
    end
  end
end

end # module
