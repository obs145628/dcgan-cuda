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

Tensor4 conv2d_sp(const Tensor4& input, const Tensor4& filter,
                  std::size_t sh, std::size_t sw,
                  std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
{
    return conv_no_pad(input.pad0(p1, p2, p3, p4), filter, sh, sw);
}


Tensor4 conv2d_sp_dk(const Tensor4& x, const Tensor4& dy,
                     std::size_t sh, std::size_t sw,
                     std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
{
    Tensor4 xtr = x.pad0(p1, p2, p3, p4).transpose(3, 1, 2, 0);
    Tensor4 f_dy = dy.transpose(1, 2, 0, 3).fstride0(sh - 1, sw - 1);
    Tensor4 o_dk = conv_no_pad(xtr, f_dy, 1, 1);
    return o_dk.transpose(1, 2, 0, 3);
}

Tensor4 conv2d_sp_dx(const Tensor4& k, const Tensor4& dy,
                     std::size_t sh, std::size_t sw,
                     std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
{
    Tensor4 pdy = dy.istride0(sh - 1, sw -1).pad0(k.d1 - 1, k.d2 - 1);
    Tensor4 k180 = k.frot180().transpose(0, 1, 3, 2);
    Tensor4 dx_full = conv_no_pad(pdy, k180, 1, 1);
    return dx_full.iregion(p1, p3, dx_full.d2 - p1 - p2, dx_full.d3 - p3 - p4);
}
