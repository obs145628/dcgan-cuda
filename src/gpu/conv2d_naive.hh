#pragma once

namespace gpu
{

    namespace
    {

        constexpr std::size_t BLOCK_SIZE = 1024;

        template <class T1, class T2>
        __device__ dbl_t compute_val(const T1& x, const T2& k,
                                     std::size_t d1, std::size_t d2,
                                     std::size_t d3, std::size_t d4,
                                     std::size_t sh, std::size_t sw)
        {
            dbl_t res = 0;
            for (std::size_t i1 = 0; i1 < k.d1(); ++i1)
                for (std::size_t i2 = 0; i2 < k.d2(); ++i2)
                    for (std::size_t i3 = 0; i3 < k.d3(); ++i3)
                    {
                        
                        dbl_t vx = t_get(x, d1, d2*sh + i1, d3*sw + i2, i3);
                        dbl_t vk = t_get(k, i1, i2, i3, d4);
                        res += vx * vk;
                    }
            return res;
        }


        template <class T1, class T2, class T3>
        __global__ void conv2d_naive(const T1 x, const T2 k, const T3 y,
                                     std::size_t sh, std::size_t sw)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= y.size())
                return;
            
            std::size_t y_i1 = index / (y.d2() * y.d3() * y.d4());
            std::size_t y_i2 = (index % (y.d2() * y.d3() * y.d4())) / (y.d3() * y.d4());
            std::size_t y_i3 = (index % (y.d3() * y.d4())) / y.d4();
            std::size_t y_i4 = index % y.d4();

            dbl_t* ptr = y(y_i1, y_i2, y_i3, y_i4);
            if (ptr)
                *ptr = compute_val(x, k,
                                   y_i1, y_i2, y_i3, y_i4, sh, sw);
        }

        void conv2d_fwd_naive(const dbl_t* x, const dbl_t* k, dbl_t* y,
                              std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                              std::size_t pad_top, std::size_t pad_left,
                              std::size_t pad_bot, std::size_t pad_right,
                              std::size_t hk, std::size_t wk, std::size_t ck,
                              std::size_t hy, std::size_t wy,
                              std::size_t sh, std::size_t sw)
        {
            Tensor4Pad<const dbl_t*> tx(x, nx, hx, wx, cx,
                                        pad_top, pad_left, pad_bot, pad_right);
            Tensor4<const dbl_t*> tk(k, hk, wk, cx, ck);
            Tensor4<dbl_t*> ty(y, nx, hy, wy, ck);

            std::size_t len = ty.size();
            std::size_t nb_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            conv2d_naive<<<nb_blocks, BLOCK_SIZE>>>(tx, tk, ty, sh, sw);
        }



        void conv2d_dx_naive(const dbl_t* k, const dbl_t* dy, dbl_t* dx,
                              std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                              std::size_t pad_top, std::size_t pad_left,
                              std::size_t pad_bot, std::size_t pad_right,
                              std::size_t hk, std::size_t wk, std::size_t ck,
                              std::size_t hy, std::size_t wy,
                              std::size_t sh, std::size_t sw)
        {
            Tensor4DxDk<const float*> tk(k, hk, wk, cx, ck);
            Tensor4DxDy<const float*> tdy(dy, nx, hy, wy, ck,
                                          hk - 1, wk - 1, hk - 1, wk - 1,
                                          sh - 1, sw - 1);
            Tensor4Pad<float*> tdx(dx, nx, hx, wx, cx,
                                   pad_top, pad_left, pad_right, pad_bot);

            std::size_t len = tdx.size();
            std::size_t nb_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

            conv2d_naive<<<nb_blocks, BLOCK_SIZE>>>(tdy, tk, tdx, 1, 1);
        }

         void conv2d_dk_naive(const dbl_t* x, const dbl_t* dy, dbl_t* dk,
                             std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                             std::size_t pad_top, std::size_t pad_left,
                             std::size_t pad_bot, std::size_t pad_right,
                             std::size_t hk, std::size_t wk, std::size_t ck,
                             std::size_t hy, std::size_t wy,
                             std::size_t sh, std::size_t sw)
        {
            Tensor4DkX<const float*> tx(x, nx, hx, wx, cx,
                                        pad_top, pad_left, pad_bot, pad_right);
            Tensor4DkDy<const float*> tdy(dy, nx, hy, wy, ck, sh - 1, sw - 1);
            Tensor4Tr3124<float*> tdk(dk, hk, wk, cx, ck);

            std::size_t len = tdk.size();
            std::size_t nb_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            conv2d_naive<<<nb_blocks, BLOCK_SIZE>>>(tx, tdy, tdk, 1, 1);
        }
    }
    
}
