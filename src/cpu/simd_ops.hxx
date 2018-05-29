#pragma once

#include "simd_kernels.hh"
#include <immintrin.h>


namespace cpu
{

    /*
    inline void simd_sigmoid(const dbl_t* a, dbl_t* out, std::size_t n)
    {
        for (std::size_t i = 0; i < n; i += 8)
        {
            __m256 srci = _mm256_load_ps(a + i);
            __m256 mx = _mm256_sub_ps(_mm256_set1_ps(0), srci);
            __m256 ex = _mm256_cexp_ps(mx);
        }
    }
    */
    
}
