module Types 

import Base: getproperty

export TilingParams

"""
  TilingParams{mc, nc, kc, mr, nr, vl} 

Handles parameters for tilings of matrices in matrix-matrix multiply.
"""
struct TilingParams{mc, nc, kc, mr, nr, vl} end

function getproperty(sizes::TilingParams{mc, nc, kc, mr, nr, vl}, sym::Symbol) where {mc, nc, kc, mr, nr, vl}
  if sym == :mc
    return mc
  elseif sym == :nc
    return nc
  elseif sym == :kc
    return kc
  elseif sym == :mr
    return mr
  elseif sym == :nr
    return nr
  elseif sym == :vl
    return vl
  else
    return getfield(sizes, name)
  end
end

end
