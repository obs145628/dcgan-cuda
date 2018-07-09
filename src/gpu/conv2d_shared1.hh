#pragma once

namespace gpu
{

    namespace
    {


        template <class T1, class T2>
        __device__ dbl_t compute_val2(const T1& x, const T2& k,
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



        /**
         * Try to use several channels
         * does not work either
        template <class T1, class T2, class T3,
                  std::size_t BlockSize,
                  std::size_t KernelSize,
                  std::size_t StrideSize,
                  std::size_t ChanOutSize>
        __global__ void conv2d_shared1(const T1 x, const T2 k, const T3 y,
                                       std::size_t nb_sub_x, std::size_t nb_sub_y)
        {

            constexpr std::size_t tilex_size = (BlockSize - 1) * StrideSize
                + KernelSize;
            constexpr std::size_t n_iters_tilex = (tilex_size + BlockSize - 1) / BlockSize;
            
            std::size_t block_x = (blockIdx.x / nb_sub_y) * BlockSize;
            std::size_t block_y = (blockIdx.x % nb_sub_y) * BlockSize;


            std::size_t thread_x = threadIdx.x / BlockSize;
            std::size_t thread_y = threadIdx.x % BlockSize;
            std::size_t thread_c = threadIdx.y;
            
            std::size_t y_row = block_x + thread_x;
            std::size_t y_col = block_y + thread_y;
            std::size_t y_i1 = blockIdx.y / (y.d4() / ChanOutSize);
            std::size_t y_i4 = (blockIdx.y % (y.d4() / ChanOutSize)) * ChanOutSize;

            bool has_val = y_row < y.d2() && y_col < y.d3();

            dbl_t vals[ChanOutSize];
            for (std::size_t i = 0; i < ChanOutSize; ++i)
                vals[i] = 0;

            std::size_t min_x = block_x * StrideSize;
            std::size_t min_y = block_y * StrideSize;

            std::size_t max_x = min_x + tilex_size; 
            std::size_t max_y = min_y + tilex_size;
            
            for (std::size_t chan = 0; chan < k.d3(); ++chan)
            {
                //store in shared memory
                __shared__ dbl_t tile_k[KernelSize][KernelSize][ChanOutSize];
                __shared__ dbl_t tile_x[tilex_size][tilex_size];

                if (thread_x < KernelSize && thread_y < KernelSize)
                {
                    dbl_t vk = t_get(k, thread_x, thread_y, chan, y_i4 + thread_c);
                    tile_k[thread_x][thread_y][thread_c] = vk;
                }

                if (thread_c == 0)
                    for (std::size_t i = 0; i < n_iters_tilex; ++i)
                        for (std::size_t j = 0; j < n_iters_tilex; ++j)
                        {
                            std::size_t id_x = min_x + BlockSize * i + thread_x;
                            std::size_t id_y = min_y + BlockSize * j + thread_y;
                            if (id_x < max_x && id_y < max_y)
                            {
                                dbl_t vx = t_get(x, y_i1, id_x, id_y, chan);
                                tile_x[id_x - min_x][id_y - min_y] = vx;
                            }
                        }

                __syncthreads();
                    
                if (has_val)
                    for (std::size_t i4 = 0; i4 < ChanOutSize; ++i4)
                        for (std::size_t i1 = 0; i1 < KernelSize; ++i1)
                            for (std::size_t i2 = 0; i2 < KernelSize; ++i2)
                            {
                                dbl_t vx = tile_x[y_row*StrideSize + i1 - min_x]
                                    [y_col*StrideSize + i2 - min_y];
                                dbl_t vk = tile_k[i1][i2][i4];
                                vals[i4] += vx * vk;
                            }

                __syncthreads();

            }

            if (has_val)
            {
                for (std::size_t i4 = 0; i4 < ChanOutSize; ++i4)
                {
                    dbl_t* ptr = y(y_i1, y_row, y_col, y_i4 + i4);
                    *ptr = vals[i4];
                }
            }
        }
        */

        template <class T1, class T2, class T3,
                  std::size_t BlockSize,
                  std::size_t KernelSize,
                  std::size_t StrideSize>
        __global__ void conv2d_shared1(const T1 x, const T2 k, const T3 y,
                                       std::size_t nb_sub_x, std::size_t nb_sub_y)
        {

            constexpr std::size_t tilex_size = (BlockSize - 1) * StrideSize
                + KernelSize;
            constexpr std::size_t n_iters_tilex = (tilex_size + BlockSize - 1) / BlockSize;
            
            std::size_t block_x = (blockIdx.x / nb_sub_y) * BlockSize;
            std::size_t block_y = (blockIdx.x % nb_sub_y) * BlockSize;

            std::size_t thread_x = threadIdx.x;
            std::size_t thread_y = threadIdx.y;
            
            std::size_t y_row = block_x + thread_x;
            std::size_t y_col = block_y + thread_y;
            std::size_t y_i1 = blockIdx.y / y.d4();
            std::size_t y_i4 = blockIdx.y % y.d4();

            bool has_val = y_row < y.d2() && y_col < y.d3();

            dbl_t val = 0;


            std::size_t min_x = block_x * StrideSize;
            std::size_t min_y = block_y * StrideSize;

            std::size_t max_x = min_x + tilex_size; 
            std::size_t max_y = min_y + tilex_size;
            
            for (std::size_t chan = 0; chan < k.d3(); ++chan)
            {
                //store in shared memory
                __shared__ dbl_t tile_k[KernelSize][KernelSize];
                __shared__ dbl_t tile_x[tilex_size][tilex_size];

                if (thread_x < KernelSize && thread_y < KernelSize)
                {
                    dbl_t vk = t_get(k, thread_x, thread_y, chan, y_i4);
                    tile_k[thread_x][thread_y] = vk;
                }

                for (std::size_t i = 0; i < n_iters_tilex; ++i)
                    for (std::size_t j = 0; j < n_iters_tilex; ++j)
                    {
                        std::size_t id_x = min_x + BlockSize * i + thread_x;
                        std::size_t id_y = min_y + BlockSize * j + thread_y;
                        if (id_x < max_x && id_y < max_y)
                        {
                            dbl_t vx = t_get(x, y_i1, id_x, id_y, chan);
                            tile_x[id_x - min_x][id_y - min_y] = vx;
                        }
                    }

                __syncthreads();
                    
                if (has_val)
                    for (std::size_t i1 = 0; i1 < KernelSize; ++i1)
                        for (std::size_t i2 = 0; i2 < KernelSize; ++i2)
                        {
                            dbl_t vx = tile_x[y_row*StrideSize + i1 - min_x]
                                [y_col*StrideSize + i2 - min_y];
                            dbl_t vk = tile_k[i1][i2];
                            val += vx * vk;
                        }

                __syncthreads();

            }

            if (has_val)
            {
                dbl_t* ptr = y(y_i1, y_row, y_col, y_i4);
                *ptr = val;
            }
        }
        

        /**
         * Try to use full image, does not work because of data representation in memory
         *
         *
        template <class T1, class T2, class T3,
                  std::size_t BlockSize,
                  std::size_t KernelSize,
                  std::size_t StrideSize,
                  std::size_t InputSize>
        __global__ void conv2d_shared1(const T1 x, const T2 k, const T3 y)
        {

            constexpr std::size_t n_iters_tilex = (InputSize + BlockSize - 1) / BlockSize;

            std::size_t y_row = threadIdx.x;
            std::size_t y_col = threadIdx.y;
            std::size_t y_i1 = blockIdx.y / y.d4();
            std::size_t y_i4 = blockIdx.y % y.d4();

            bool has_val = y_row < y.d2() && y_col < y.d3();

            dbl_t val = 0;
            
            for (std::size_t chan = 0; chan < k.d3(); ++chan)
            {
                //store in shared memory
                __shared__ dbl_t tile_k[KernelSize][KernelSize];
                __shared__ dbl_t tile_x[InputSize][InputSize];


                if (threadIdx.x < KernelSize && threadIdx.y < KernelSize)
                {
                    dbl_t vk = t_get(k, threadIdx.x, threadIdx.y, chan, y_i4);
                    tile_k[threadIdx.x][threadIdx.y] = vk;
                }

                for (std::size_t i = 0; i < n_iters_tilex; ++i)
                {
                    for (std::size_t j = 0; j < n_iters_tilex; ++j)
                    {
                        std::size_t id_x = BlockSize * i + threadIdx.x;
                        std::size_t id_y = BlockSize * j + threadIdx.y;
                        if (id_x < InputSize && id_y < InputSize)
                        {
                            dbl_t vx = t_get(x, y_i1, id_x, id_y, chan);
                            tile_x[id_x][id_y] = vx;
                        }
                    }
                }

                __syncthreads();
                    
                if (has_val)
                    for (std::size_t i1 = 0; i1 < KernelSize; ++i1)
                        for (std::size_t i2 = 0; i2 < KernelSize; ++i2)
                        {
                            dbl_t vx = tile_x[y_row*StrideSize + i1][y_col*StrideSize + i2];
                            dbl_t vk = tile_k[i1][i2];
                            val += vx * vk;
                        }

                __syncthreads();

            }

            if (has_val)
            {
                dbl_t* ptr = y(y_i1, y_row, y_col, y_i4);
                *ptr = val;
            }
        }
        */
         
        void conv2d_fwd_shared1(const dbl_t* x, const dbl_t* k, dbl_t* y,
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

            constexpr std::size_t block_size = 8;
            constexpr std::size_t kernel_size = 5;
            constexpr std::size_t stride_size = 2;

            std::size_t suby_rows = block_size;
            std::size_t suby_cols = block_size;

            std::size_t nb_sub_x = (ty.d2() + suby_rows - 1) / suby_rows; 
            std::size_t nb_sub_y = (ty.d3() + suby_cols - 1) / suby_cols;

            dim3 threads_per_block(suby_rows, suby_cols);
            dim3 blocks_per_grid(nb_sub_x * nb_sub_y, ty.d1() * ty.d4());

            using T1 = decltype(tx);
            using T2 = decltype(tk);
            using T3 = decltype(ty);


            conv2d_shared1<T1, T2, T3,
                           block_size,
                           kernel_size,
                           stride_size><<<blocks_per_grid, threads_per_block>>>(tx, tk, ty,
                                                                               nb_sub_x, nb_sub_y);
        }
        
    }
    
}
