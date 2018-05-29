#pragma once

#include "../memory/types.hh"
#include <cstddef>

namespace cpu
{

    /**
     * Perform sigmoid operation of a vector
     * out = sigmoid(a)
     * a - vector (n)
     * out - vector (n)
     */
    void simd_sigmoid(const dbl_t* a, dbl_t* out, std::size_t n);

}

#include "simd_ops.hxx"
