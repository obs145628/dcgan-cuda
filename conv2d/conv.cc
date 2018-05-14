#include "conv.hh"

#include <cassert>
#include <iostream>

float compute_val(const Tensor4& input, const Tensor4& filter,
                  std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4)
{
    float res = 0;
    for (std::size_t i1 = 0; i1 < filter.d1; ++i1)
        for (std::size_t i2 = 0; i2 < filter.d2; ++i2)
            for (std::size_t i3 = 0; i3 < filter.d3; ++i3)
                res += input(d1, d2 + i1, d3 + i2, i3) * filter(i1, i2, i3, d4);
    return res;
}


Tensor4 conv_no_pad(const Tensor4& input, const Tensor4& filter,
                    std::size_t sh, std::size_t sw)
{
    std::size_t h = (input.d2 - filter.d1) / sh + 1;
    std::size_t w = (input.d3 - filter.d2) / sw + 1;

        
    Tensor4 out(input.d1, h, w, filter.d4);

    for (std::size_t i1 = 0; i1 < input.d1; ++i1)
        for (std::size_t i2 = 0; i2 < h; ++i2)
            for (std::size_t i3 = 0; i3 < w; ++i3)
                for (std::size_t i4 = 0; i4 < filter.d4; ++i4)
                    out(i1, i2, i3, i4) = compute_val(input, filter, i1, i2 * sh, i3 * sw, i4);

    return out;
}

Tensor4 conv_pad(const Tensor4& input, const Tensor4& filter,
                 std::size_t sh, std::size_t sw,
                 std::size_t ph, std::size_t pw)
{
    return conv_no_pad(input.pad0(ph, pw), filter, sh, sw);
}
