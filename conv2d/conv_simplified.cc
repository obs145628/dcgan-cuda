#include "conv_simplified.hh"

namespace
{

    /**
     * Suppose x (h, w, 1, 1)
     * y = rot180(x)
     * y (h, w, 1, 1)
     */
    Tensor4 rot180(Tensor4 x)
    {
        std::size_t h = x.d1;
        std::size_t w = x.d2;
        Tensor4 res(h, w, 1, 1);
        for (std::size_t i = 0; i < h; ++i)
            for (std::size_t j = 0; j < w; ++j)
                res(i, j, 0, 0) = x(h - i - 1, w - j - 1, 0, 0);
        return res;
    }
    
}


Tensor4 conv_c1f1i1s1p0(Tensor4 x, Tensor4 k)
{
    std::size_t hx = x.d2;
    std::size_t wx = x.d3;
    std::size_t hk = k.d1;
    std::size_t wk = k.d2;
    std::size_t hy = hx - hk + 1;
    std::size_t wy = wx - wk + 1;
    Tensor4 y(1, hy, wy, 1);

    for (std::size_t i = 0; i < hy; ++i)
        for (std::size_t j = 0; j < wy; ++j)
        {
            float s = 0;
            for (std::size_t ik = 0; ik < hk; ++ik)
                for (std::size_t jk = 0; jk < wk; ++jk)
                    s += x(0, i + ik, j + jk, 0) * k(ik, jk, 0, 0);
            y(0, i, j, 0) = s;
        }

    return y;
}

Tensor4 conv_dk_c1f1i1s1p0(Tensor4 x, Tensor4 dy)
{
    Tensor4 f_dy = dy.reshape(dy.d2, dy.d3, 1, 1);
    Tensor4 o_dk = conv_c1f1i1s1p0(x, f_dy);
    return o_dk.reshape(o_dk.d2, o_dk.d3, 1, 1);
}

Tensor4 conv_dx_c1f1i1s1p0(Tensor4 k, Tensor4 dy)
{
    Tensor4 pdy = dy.pad0(k.d1 - 1, k.d2 - 1);
    Tensor4 k180 = rot180(k);
    return conv_c1f1i1s1p0(pdy, k180);
}
