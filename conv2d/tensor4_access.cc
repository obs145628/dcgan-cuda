#include "tensor4_access.hh"

namespace acc
{

    namespace
    {

        template <class T1, class T2>
        float compute_val(T1& tx, T2& tk,
                          std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4)
        {
            float res = 0;
            for (std::size_t i1 = 0; i1 < tk.d1(); ++i1)
                for (std::size_t i2 = 0; i2 < tk.d2(); ++i2)
                    for (std::size_t i3 = 0; i3 < tk.d3(); ++i3)
                    {
                        float vx = t_get(tx, d1, d2 + i1, d3 + i2, i3);
                        float vk = t_get(tk, i1, i2, i3, d4);
                        res += vx * vk;
                    }
            return res;
        }


        template <class T1, class T2, class T3>
        void conv_no_pad(T1& tx, T2& tk, T3& ty,
                         std::size_t sh, std::size_t sw)
        {

            for (std::size_t i1 = 0; i1 < ty.d1(); ++i1)
                for (std::size_t i2 = 0; i2 < ty.d2(); ++i2)
                    for (std::size_t i3 = 0; i3 < ty.d3(); ++i3)
                        for (std::size_t i4 = 0; i4 < ty.d4(); ++i4)
                        {
                            float* ptr = ty(i1, i2, i3, i4);
                            if (ptr)
                                *ptr = compute_val(tx, tk,
                                                   i1, i2 * sh, i3 * sw, i4);
                        }

        }
        
    }
    

    /**
     * Compute a convolution with padding and strides
     * input (i1, h1, w1, c1) nb input * input height * input width * input channels
     * filter (fh, fw, c1, k) filter height * filter with * input channels * nb filters
     * sh - height of strides
     * sw - width of strides
     * p1 - top padding
     * p2 - bottom padding
     * p3 - left padding
     * p4 - right padding
     * out (i1, (h1 - fh + p1 + p2) / sh + 1, (w1 - fw + p3 + p4) / sw + 1, k)
     */
    ::Tensor4 conv2d_sp(const ::Tensor4& input, const ::Tensor4& filter,
                      std::size_t sh, std::size_t sw,
                      std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
    {

        std::size_t x_fh = input.d2 + p1 + p2;
        std::size_t x_fw = input.d3 + p3 + p4;
        
        std::size_t y_h = (x_fh - filter.d1) / sh + 1;
        std::size_t y_w = (x_fw - filter.d2) / sw + 1;

        
        ::Tensor4 out(input.d1, y_h, y_w, filter.d4);

        conv2d_sp(input.data, filter.data, out.data,
                  input.d1, input.d2, input.d3, input.d4,
                  filter.d1, filter.d2, filter.d4,
                  sh, sw, p1, p2, p3, p4);
        
        return out;
    }

    void conv2d_sp(const float* x, const float* k, float* y,
                   std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                   std::size_t hk, std::size_t wk, std::size_t ck,
                   std::size_t sh, std::size_t sw,
                   std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
    {
        std::size_t hy = (hx + p1 + p2 - hk) / sh + 1;
        std::size_t wy = (wx + p3 + p4 - wk) / sw + 1;
        
        Tensor4Pad<const float*> tx(x, nx, hx, wx, cx, p1, p3, p2, p4);
        Tensor4<const float*> tk(k, hk, wk, cx, ck);
        Tensor4<float*> ty(y, nx, hy, wy, ck);

        conv_no_pad(tx, tk, ty, sh, sw);
    }

    ::Tensor4 conv2d_sp_dk(const ::Tensor4& input, const ::Tensor4& dout,
                           std::size_t sh, std::size_t sw,
                           std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
    {

        std::size_t hk = input.d2 + p1 + p2 - sh * (dout.d2 - 1);
        std::size_t wk = input.d3 + p3 + p4 - sw * (dout.d3 - 1);

        ::Tensor4 dfilter(hk, wk, input.d4, dout.d4);

        conv2d_sp_dk(input.data, dout.data, dfilter.data,
                     input.d1, input.d2, input.d3, input.d4,
                     dout.d2, dout.d3, dout.d4,
                     sh, sw, p1, p2, p3, p4);

        return dfilter;
    }

    void conv2d_sp_dk(const float* x, const float* dy, float* dk,
                      std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                      std::size_t hy, std::size_t wy, std::size_t cy,
                      std::size_t sh, std::size_t sw,
                      std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
    {
        std::size_t hk = hx + p1 + p2 - sh * (hy - 1);
        std::size_t wk = wx + p3 + p4 - sw * (wy - 1);

        Tensor4DkX<const float*> tx(x, nx, hx, wx, cx, p1, p3, p2, p4);
        Tensor4DkDy<const float*> tdy(dy, nx, hy, wy, cy, sh - 1, sw - 1);
        Tensor4Tr3124<float*> tdk(dk, hk, wk, cx, cy);
        conv_no_pad(tx, tdy, tdk, 1, 1);
    }

    ::Tensor4 conv2d_sp_dx(const ::Tensor4& filter, const ::Tensor4& dout,
                           std::size_t sh, std::size_t sw,
                           std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
    {
        std::size_t hx = sh * (dout.d2 - 1) + filter.d1 - p1 - p2;
        std::size_t wx = sw * (dout.d3 - 1) + filter.d2 - p3 - p4;
        
        ::Tensor4 dinput(dout.d1, hx, wx, filter.d3);

        conv2d_sp_dx(filter.data, dout.data, dinput.data,
                     filter.d1, filter.d2, filter.d3,
                     dout.d1, dout.d2, dout.d3, dout.d4,
                     sh, sw, p1, p2, p3, p4);
        return dinput;
    }

    void conv2d_sp_dx(const float* k, const float* dy, float* dx,
                      std::size_t hk, std::size_t wk, std::size_t ck,
                      std::size_t ny, std::size_t hy, std::size_t wy, std::size_t cy,
                      std::size_t sh, std::size_t sw,
                      std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4)
    {
        std::size_t hx = sh * (hy - 1) + hk - p1 - p2;
        std::size_t wx = sw * (wy - 1) + wk - p3 - p4;

        Tensor4DxDk<const float*> tk(k, hk, wk, ck, cy);
        Tensor4DxDy<const float*> tdy(dy, ny, hy, wy, cy,
                                      hk - 1, wk - 1, hk - 1, wk - 1,
                                      sh - 1, sw - 1);
        Tensor4Pad<float*> tdx(dx, ny, hx, wx, ck, p1, p3, p2, p4);
        conv_no_pad(tdy, tk, tdx, 1, 1);
    }
    
}
