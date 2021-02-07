module TransposeNoPacking

using KernelAbstractions.Extras: @unroll
using SIMD: Vec, vload, vstore
using StaticArrays: MArray, MVector
using MyGEMM.Types: TilingParams

export multiply_transpose_no_packing!

"""
  multiply_transpose_no_packing!(C, A, B, params)

Performs `C += A * B'` without packing optimization.
"""
@inline function multiply_transpose_no_packing!(C, A, B, params::TilingParams) 
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

  # We assume that the matrices are even divisible by `nr` and `mr`, 
  # that the blocking perfectly partitions the matrix,
  # and, vectors are perfectly divided by the chosen vector unit
  @assert mod(m, mr)  == 0
  @assert mod(n, nr)  == 0
  @assert mod(mr, vl) == 0

  # Execute C = C + A * B'
  loop5!(C, A, B, params)
end

@inline function loop5!(C, A, B, params::TilingParams)
  n = size(C, 2)
  nc = params.nc
  @inbounds for j1 = 1:nc:n
    j2 = min(n, j1 + nc - 1)
    C_j = @view C[:, j1:j2]
    B_j = @view B[j1:j2, :]
    loop4!(C_j, A, B_j, params)
  end
end

@inline function loop4!(C, A, B, params ::TilingParams)
  k = size(A, 2)
  kc = params.kc
  @inbounds for p1 = 1:kc:k
    p2 = min(k, p1 + kc - 1)
    A_p = @view A[:, p1:p2]
    B_p = @view B[:, p1:p2]
    loop3!(C, A_p, B_p, params
          )
  end
end

@inline function loop3!(C, A, B, params::TilingParams)
  m = size(C, 1)
  mc = params.mc
  @inbounds for i1 = 1:mc:m
    i2 = min(m, i1 + mc - 1)
    C_i = @view C[i1:i2, :]
    A_i = @view A[i1:i2, :]
    loop2!(C_i, A_i, B, params)
  end
end

@inline function loop2!(C, A, B, params::TilingParams)
  n = size(C, 2)
  nr = params.nr
  @inbounds for j1 = 1:nr:n
    j2 = min(n, j1 + nr - 1)
    C_j = @view C[:, j1:j2]
    B_j = @view B[j1:j2, :]
    loop1!(C_j, A, B_j, params
          )
  end
end

@inline function loop1!(C, A, B, params::TilingParams)
  m = size(C, 1)
  mr = params.mr
  @inbounds for i1 = 1:mr:m
    i2 = min(m, i1 + mr - 1)
    C_i = @view C[i1:i2, :]
    A_i = @view A[i1:i2, :]
    microknl!(C_i, A_i, B, params
      )
  end
end

@inline function microknl!(C, A, B, params::TilingParams)
  # Get relevant tile params
  m     = size(parent(C), 1)
  mr    = params.mr
  nr    = params.nr
  vl    = params.vl
  mr_vl = div(mr, vl)
  pend  = size(B, 2)

  # Set relevant types
  FT = eltype(C)
  VT = Vec{vl, FT}
 
  # Initialize vector storage
  C_tile  = MArray{Tuple{mr_vl, nr}, VT}(undef)
  A_panel = MVector{mr_vl, VT}(undef)
  
  # Load the columns of the microtile of C
  @inbounds @unroll for j = 1:nr
    @unroll for i = 1:mr_vl
      offset = ((i - 1) * vl + (j - 1) * m) * sizeof(FT)
      C_tile[i, j] = vload(VT, pointer(C) + offset)
    end
  end

  # Do the rank-one updates
  @inbounds @unroll for p = 1:pend
    # Load pieces of `A_panel`
    @unroll for i = 1:mr_vl
      offset = ((p - 1) * m + (i - 1) * vl) * sizeof(FT)
      A_panel[i] = vload(VT, pointer(A) + offset)
    end
    
    # Do outer product updates
    @unroll for j = 1:nr
      β = B[j, p]
      @unroll for i = 1:mr_vl
        α = A_panel[i]
        C_tile[i, j] = muladd(α, β, C_tile[i, j])
      end
    end
  end

  # Write back from vector storage
  @inbounds @unroll for j = 1:nr
    @unroll for i = 1:mr_vl
      offset = ((i - 1) * vl + (j - 1) * m) * sizeof(FT)
      vstore(C_tile[i, j], pointer(C) + offset)
    end
  end
end

end
