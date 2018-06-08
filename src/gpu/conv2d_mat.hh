#pragma once

namespace gpu
{

    namespace
    {

        constexpr std::size_t MAT_BLOCK_SIZE = 512;

        template <class T>
        __device__ const dbl_t* kernel_mat_elem(const T& k, std::size_t i, std::size_t j)
        {

            //i: index of the output element of y
            //j: index of the element of the dot-product to compute

            //std::size_t y_i1 = i / (y.d2() * y.d3() * y.d4());
            //std::size_t y_i2 = (i % (y.d2() * y.d3() * y.d4())) / (y.d3() * y.d4());
            //std::size_t y_i3 = (i % (y.d3() * y.d4())) / y.d4();

            std::size_t k_i4 = i % k.d4();

            std::size_t k_i1 = j / (k.d2() * k.d3());
            std::size_t k_i2 = (j % k.d2() * k.d3()) / k.d3();
            std::size_t k_i3 = j % k.d3();
            
            
            return k(k_i1, k_i2, k_i3, k_i4);
        }

        template <class T1, class T2, class T3>
        __device__ const dbl_t* input_mat_elem(const T1& x, const T2& k, const T3& y,
                                               std::size_t i, std::size_t j, std::size_t j2,
                                               std::size_t sh, std::size_t sw)
        {

            //i: index of the element of the dot-product to compute
            //j: index of the image
            //j2: index of the output element of y

            std::size_t k_i1 = i / (k.d2() * k.d3());
            std::size_t k_i2 = (i % k.d2() * k.d3()) / k.d3();
            std::size_t k_i3 = i % k.d3();

            std::size_t y_i1 = j;
            std::size_t y_i2 = (j2 % (y.d2() * y.d3() * y.d4())) / (y.d3() * y.d4());
            std::size_t y_i3 = (j2 % (y.d3() * y.d4())) / y.d4();
            //std::size_t y_i4 = j2 % y.d4();

            return x(y_i1, y_i2 * sh + k_i1, y_i3 * sw + k_i2, k_i3);
        }


        template <class T>
        __device__ dbl_t* output_mat_elem(const T& y, std::size_t i, std::size_t j)
        {
            //i : index of the output element of y to compute
            //j: index of the image


            std::size_t y_i1 = j;
            std::size_t y_i2 = (i % (y.d2() * y.d3() * y.d4())) / (y.d3() * y.d4());
            std::size_t y_i3 = (i % (y.d3() * y.d4())) / y.d4();
            std::size_t y_i4 = i % y.d4();
            return y(y_i1, y_i2, y_i3, y_i4);
        }

        template <class T1, class T2, class T3>
        __global__ void conv2d_mat(const T1 x, const T2 k, const T3 y,
                                   std::size_t sh, std::size_t sw)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= y.size())
                return;

            std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t col = blockIdx.y * blockDim.y + threadIdx.y;

            std::size_t y_len = y.d2() * y.d3() * y.d4();
            std::size_t y_imgs = y.d1();
            std::size_t k_len = k.d1() * k.d2() * k.d3();

            if (row >= y_len || col >= y_imgs)
                return;

            dbl_t val = 0;
            for (std::size_t i = 0; i < k_len; ++i)
            {
                const dbl_t* kp = kernel_mat_elem(k, row, i);
                const dbl_t* xp = input_mat_elem(x, k, y, i, col, row, sh, sw);
                dbl_t kv = kp ? *kp : 0;
                dbl_t xv = xp ? *xp : 0;
                val += kv * xv;
            }

            dbl_t* ptr = output_mat_elem(y, row, col);
            *ptr = val;
        }



        void conv2d_fwd_mat(const dbl_t* x, const dbl_t* k, dbl_t* y,
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


            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);


            std::size_t rows = ty.d2() * ty.d3() * ty.d4();
            std::size_t cols = ty.d1();
            std::size_t block_size = 32;
            dim3 threads_per_block (block_size, block_size);
            std::size_t nb_blocks_x = (rows + block_size - 1) / block_size;
            std::size_t nb_blocks_y = (cols + block_size - 1) / block_size;
            dim3 blocks_per_grid (nb_blocks_x, nb_blocks_y);
            conv2d_mat<<<blocks_per_grid, threads_per_block>>>(tx, tk, ty, sh, sw);        
            
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float time;
            cudaEventElapsedTime(&time, start, stop);

            std::cout << "time (fwd_mat_conv) = " << time << "ms\n" << std::endl;
        }
        
    }
    
}
