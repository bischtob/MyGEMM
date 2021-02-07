module Utils

using KernelAbstractions.Extras: @unroll
using SIMD: @simd
using MyGEMM.Types: TilingParams

export packA!
export packB!

"""
  packA!(A_packed, A, params)

Packs `A` of `C += A * B` into contiguous-in-memory vector `A_packed`.
"""
@inline function packA!(A_packed, A, params::TilingParams)
  m, k = size(A)
  mr = params.mr
  # Loop through A in access order to store in packed array
  liner_ix = 1
  @inbounds for x = 1:mr:m        # which tile * mr
    @unroll 4 for p in 1:k        # which row
      @simd ivdep for y = 0:mr-1  # micro vector element
        A_packed[liner_ix] = A[x + y, p]
        liner_ix += 1
      end
    end
  end
end

"""
  packB!(B_packed, B, params)

Packs `B` of `C += A * B'` into contiguous-in-memory vector `B_packed`.
"""
@inline function packB!(B_packed, B, params::TilingParams)
  n, k = size(B)
  nr = params.nr
  # Loop through B in access order to store in packed array
  liner_ix = 1
  @inbounds for x = 1:nr:n        # which tile * nr
    @unroll 4 for p in 1:k        # which row
      @simd ivdep for y = 0:nr-1  # micro vector element
        B_packed[liner_ix] = B[x + y, p]
        liner_ix += 1
      end
    end
  end
end

end
