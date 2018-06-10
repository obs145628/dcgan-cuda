#pragma once

#include <cstdint>

namespace xorshift
{

    void seed(std::uint64_t s);
    std::uint64_t next_u64();
    float next_f32();

    void fill(float* begin, float* end);
}
