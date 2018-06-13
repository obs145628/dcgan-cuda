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
                            float val = compute_val(tx, tk,
                                                    i1, i2 * sh, i3 * sw, i4);
                            t_set(ty, i1, i2, i3, i4, val);
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
    
}
