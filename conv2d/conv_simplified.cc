#include "conv_simplified.hh"
#include "conv.hh"

#include <iostream>
#include <stdexcept>

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

    Tensor4 frot180(Tensor4 x)
    {
        Tensor4 res(x.d1, x.d2, x.d3, x.d4);
        for (std::size_t i1 = 0; i1 < x.d1; ++i1)
            for (std::size_t i2 = 0; i2 < x.d2; ++i2)
                for (std::size_t i3 = 0; i3 < x.d3; ++i3)
                    for (std::size_t i4 = 0; i4 < x.d4; ++i4)
                        res(i1, i2, i3, i4) = x(x.d1 - i1 - 1, x.d2 - i2 - 1, i3, i4);
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






Tensor4 conv_cnf1i1s1p0(Tensor4 x, Tensor4 k)
{
    std::size_t hx = x.d2;
    std::size_t wx = x.d3;
    std::size_t hk = k.d1;
    std::size_t wk = k.d2;
    std::size_t hy = hx - hk + 1;
    std::size_t wy = wx - wk + 1;
    std::size_t c = x.d4;
    Tensor4 y(1, hy, wy, 1);

    for (std::size_t i = 0; i < hy; ++i)
        for (std::size_t j = 0; j < wy; ++j)
        {
            float s = 0;
            for (std::size_t c1 = 0; c1 < c; ++c1)
                for (std::size_t ik = 0; ik < hk; ++ik)
                    for (std::size_t jk = 0; jk < wk; ++jk)
                        s += x(0, i + ik, j + jk, c1) * k(ik, jk, c1, 0);
            y(0, i, j, 0) = s;
        }

    return y;
}

Tensor4 conv_dk_cnf1i1s1p0(Tensor4 x, Tensor4 dy)
{
    Tensor4 xtr = x.transpose(3, 1, 2, 0);
    Tensor4 f_dy = dy.reshape(dy.d2, dy.d3, 1, 1);
    Tensor4 o_dk = conv_no_pad(xtr, f_dy, 1, 1);
    return o_dk.transpose(1, 2, 0, 3);
}

Tensor4 conv_dx_cnf1i1s1p0(Tensor4 k, Tensor4 dy)
{
    Tensor4 pdy = dy.pad0(k.d1 - 1, k.d2 - 1);
    Tensor4 k180 = frot180(k).reshape(k.d1, k.d2, 1, k.d3);
    return conv_no_pad(pdy, k180, 1, 1);
}









Tensor4 conv_cnfni1s1p0(Tensor4 x, Tensor4 k)
{
    std::size_t hx = x.d2;
    std::size_t wx = x.d3;
    std::size_t hk = k.d1;
    std::size_t wk = k.d2;
    std::size_t hy = hx - hk + 1;
    std::size_t wy = wx - wk + 1;
    std::size_t c = x.d4;
    std::size_t f = k.d4;
    Tensor4 y(1, hy, wy, f);

    for (std::size_t i = 0; i < hy; ++i)
        for (std::size_t j = 0; j < wy; ++j)
            for (std::size_t co = 0; co < f; ++co)
            {
                float s = 0;
                for (std::size_t c1 = 0; c1 < c; ++c1)
                    for (std::size_t ik = 0; ik < hk; ++ik)
                        for (std::size_t jk = 0; jk < wk; ++jk)
                            s += x(0, i + ik, j + jk, c1) * k(ik, jk, c1, co);
                y(0, i, j, co) = s;
            }

    return y;
}

Tensor4 conv_dk_cnfni1s1p0(Tensor4 x, Tensor4 dy)
{
    Tensor4 xtr = x.transpose(3, 1, 2, 0);
    Tensor4 f_dy = dy.reshape(dy.d2, dy.d3, 1, dy.d4);
    Tensor4 o_dk = conv_no_pad(xtr, f_dy, 1, 1);
    return o_dk.transpose(1, 2, 0, 3);
}

Tensor4 conv_dx_cnfni1s1p0(Tensor4 k, Tensor4 dy)
{
    Tensor4 pdy = dy.pad0(k.d1 - 1, k.d2 - 1);
    Tensor4 k180 = frot180(k).transpose(0, 1, 3, 2);
    return conv_no_pad(pdy, k180, 1, 1);
}















Tensor4 conv_c1f1i1snp0(Tensor4 x, Tensor4 k, std::size_t sh, std::size_t sw)
{
    std::size_t hx = x.d2;
    std::size_t wx = x.d3;
    std::size_t hk = k.d1;
    std::size_t wk = k.d2;
    std::size_t hy = (hx - hk) / sh + 1;
    std::size_t wy = (wx - wk) / sw + 1;
    Tensor4 y(1, hy, wy, 1);

    if ((hy - 1) * sh != hx - hk)
        throw std::runtime_error {"Invalid height"};
    if ((wy - 1) * sw != wx - wk)
        throw std::runtime_error {"Invalid width"};

    for (std::size_t i = 0; i < hy; ++i)
        for (std::size_t j = 0; j < wy; ++j)
        {
            float s = 0;
            for (std::size_t ik = 0; ik < hk; ++ik)
                for (std::size_t jk = 0; jk < wk; ++jk)
                    s += x(0, i*sh + ik, j*sw + jk, 0) * k(ik, jk, 0, 0);
            y(0, i, j, 0) = s;
        }

    return y;
}

Tensor4 conv_dk_c1f1i1snp0(Tensor4 x, Tensor4 dy, std::size_t sh, std::size_t sw)
{
    Tensor4 f_dy = dy.reshape(dy.d2, dy.d3, 1, 1).fstride0(sh - 1, sw - 1);
    Tensor4 o_dk = conv_c1f1i1s1p0(x, f_dy);
    return o_dk.reshape(o_dk.d2, o_dk.d3, 1, 1);
}

Tensor4 conv_dx_c1f1i1snp0(Tensor4 k, Tensor4 dy, std::size_t sh, std::size_t sw)
{
    Tensor4 pdy = dy.istride0(sh - 1, sw -1).pad0(k.d1 - 1, k.d2 - 1);
    //Tensor4 pdy = dy.pad0(k.d1 - 1, k.d2 - 1);
    Tensor4 k180 = rot180(k);
    return conv_c1f1i1s1p0(pdy, k180);
}








Tensor4 conv_c1f1i1snpn(Tensor4 x, Tensor4 k, std::size_t sh, std::size_t sw,
                        std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
{
    return conv_c1f1i1snp0(x.pad0(p1, p2, p3, p4), k, sh, sw);
}


Tensor4 conv_dk_c1f1i1snpn(Tensor4 x, Tensor4 dy, std::size_t sh, std::size_t sw,
                           std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
{
    return conv_dk_c1f1i1snp0(x.pad0(p1, p2, p3, p4), dy, sh, sw);
}

Tensor4 conv_dx_c1f1i1snpn(Tensor4 k, Tensor4 dy, std::size_t sh, std::size_t sw,
                           std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
{
    
    Tensor4 dx_full = conv_dx_c1f1i1snp0(k, dy, sh, sw);
    return dx_full.iregion(p1, p3, dx_full.d2 - p1 - p2, dx_full.d3 - p3 - p4);
}
