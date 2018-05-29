#pragma once

#include <cstddef>
#include "fwd.hh"

namespace cpu
{

    constexpr std::size_t KERNEL_SIMD_OFFSET = 300;
    
    extern kernel_f kernels_list[512];

    void kernels_init();
}
