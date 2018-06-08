#pragma once

#include <fstream>

namespace gpu
{

    namespace
    {

        __global__ void prepare_input(Tensor4<const dbl_t*> in,
                                      Tensor4<dbl_t*> out,
                                      std::size_t pad_top, std::size_t pad_left)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= in.size())
                return;

            std::size_t i1 = index / (in.d2() * in.d3() * in.d4());
            std::size_t i2 = (index % (in.d2() * in.d3() * in.d4())) / (in.d3() * in.d4());
            std::size_t i3 = (index % (in.d3() * in.d4())) / in.d4();
            std::size_t i4 = index % in.d4();

            const dbl_t* iptr = in(i1, i2, i3, i4);
            //dbl_t* optr = out(i1, i2 + pad_top, i3 + pad_left, i4);
            dbl_t* optr = out(i1, i4, i2 + pad_top, i3 + pad_left);
            *optr = *iptr;
        }

        __global__ void prepare_kernel(Tensor4<const dbl_t*> in,
                                       Tensor4<dbl_t*> out)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= in.size())
                return;

            std::size_t i1 = index / (in.d2() * in.d3() * in.d4());
            std::size_t i2 = (index % (in.d2() * in.d3() * in.d4())) / (in.d3() * in.d4());
            std::size_t i3 = (index % (in.d3() * in.d4())) / in.d4();
            std::size_t i4 = index % in.d4();

            const dbl_t* iptr = in(i1, i2, i3, i4);
            //dbl_t* optr = out(i1, i2, i3, i4);
            dbl_t* optr = out(i3, i4, i1, i2);
            *optr = *iptr;
        }




        /*

        template <class T1, class T2, class T3,
                  std::size_t BlockSize,
                  std::size_t KernelSize,
                  std::size_t StrideSize,
                  std::size_t InputSize>
        __global__ void conv2d_shared2(const T1 x, const T2 k, const T3 y)
        {

            constexpr std::size_t n_iters_tilex = (InputSize + BlockSize - 1) / BlockSize;
            constexpr std::size_t n_iters_tilek = (KernelSize + BlockSize - 1) / BlockSize;

            std::size_t y_row = threadIdx.x / BlockSize;
            std::size_t y_col = threadIdx.x % BlockSize;
            std::size_t y_i1 = blockIdx.x;
            std::size_t y_i4 = blockIdx.y;

            dbl_t val = 0;
            
            for (std::size_t chan = 0; chan < k.d3(); ++chan)
            {
                //store in shared memory
                __shared__ dbl_t tile_k[KernelSize][KernelSize];
                __shared__ dbl_t tile_x[InputSize][InputSize];


                for (std::size_t i = 0; i < n_iters_tilek; ++i)
                    for (std::size_t j = 0; j < n_iters_tilek; ++j)
                    {
                        std::size_t id_x = y_row + i * BlockSize;
                        std::size_t id_y = y_col + j * BlockSize;
                        if (id_x < KernelSize && id_y < KernelSize)
                        {
                            dbl_t vk = t_get(k, id_x, id_y, chan, y_i4);
                            tile_k[id_x][id_y] = vk;
                        }
                    }


                for (std::size_t i = 0; i < n_iters_tilex; ++i)
                    for (std::size_t j = 0; j < n_iters_tilex; ++j)
                    {
                        std::size_t id_x = y_row + i * BlockSize;
                        std::size_t id_y = y_col + j * BlockSize;
                        if (id_x < InputSize && id_y < InputSize)
                        {
                            dbl_t vx = t_get(x, y_i1, chan, id_x, id_y);
                            tile_x[id_x][id_y] = vx;
                        }
                    }

                __syncthreads();
                    
                for (std::size_t i1 = 0; i1 < KernelSize; ++i1)
                    for (std::size_t i2 = 0; i2 < KernelSize; ++i2)
                    {
                        dbl_t vx = tile_x[y_row*StrideSize + i1][y_col*StrideSize + i2];
                        dbl_t vk = tile_k[i1][i2];
                        val += vx * vk;
                    }

                __syncthreads();
            }

            dbl_t* ptr = y(y_i1, y_row, y_col, y_i4);
            *ptr = val;
        }

        */

        /**
         * Test kernel with several chan out elements computed by each block
        
        template <class T1, class T2, class T3,
                  std::size_t BlockSize,
                  std::size_t KernelSize,
                  std::size_t StrideSize,
                  std::size_t InputSize,
                  std::size_t ChanOutSize>
        __global__ void conv2d_shared2(const T1 x, const T2 k, const T3 y)
        {

            constexpr std::size_t n_iters_tilex = (InputSize + BlockSize - 1) / BlockSize;
            constexpr std::size_t n_iters_tilek = (KernelSize + BlockSize - 1) / BlockSize;

            std::size_t y_row = threadIdx.x / BlockSize;
            std::size_t y_col = threadIdx.x % BlockSize;
            std::size_t y_c = threadIdx.y;
            
            std::size_t y_i1 = blockIdx.x;
            std::size_t y_i4 = blockIdx.y * ChanOutSize + y_c;

            dbl_t val = 0;
            
            for (std::size_t chan = 0; chan < k.d1(); ++chan)
            {
                //store in shared memory
                __shared__ dbl_t tile_k[KernelSize][KernelSize][ChanOutSize];
                __shared__ dbl_t tile_x[InputSize][InputSize];


                for (std::size_t i = 0; i < n_iters_tilek; ++i)
                    for (std::size_t j = 0; j < n_iters_tilek; ++j)
                    {
                        std::size_t id_x = y_row + i * BlockSize;
                        std::size_t id_y = y_col + j * BlockSize;
                        if (id_x < KernelSize && id_y < KernelSize)
                        {
                            //dbl_t vk = t_get(k, id_x, id_y, chan, y_i4);
                            dbl_t vk = t_get(k, chan, y_i4, id_x, id_y);
                            tile_k[id_x][id_y][y_c] = vk;
                        }
                    }


                if (y_c == 0)
                    for (std::size_t i = 0; i < n_iters_tilex; ++i)
                        for (std::size_t j = 0; j < n_iters_tilex; ++j)
                        {
                            std::size_t id_x = y_row + i * BlockSize;
                            std::size_t id_y = y_col + j * BlockSize;
                            if (id_x < InputSize && id_y < InputSize)
                            {
                                dbl_t vx = t_get(x, y_i1, chan, id_x, id_y);
                                tile_x[id_x][id_y] = vx;
                            }
                        }

                __syncthreads();
                    
                for (std::size_t i1 = 0; i1 < KernelSize; ++i1)
                    for (std::size_t i2 = 0; i2 < KernelSize; ++i2)
                    {
                        dbl_t vx = tile_x[y_row*StrideSize + i1][y_col*StrideSize + i2];
                        dbl_t vk = tile_k[i1][i2][y_c];
                        val += vx * vk;
                    }

                __syncthreads();
            }

            dbl_t* ptr = y(y_i1, y_row, y_col, y_i4);
            *ptr = val;
        }

        */

        template <int WarpSize>
        __inline__ __device__
        dbl_t warp_reduce_sum(dbl_t val)
        {
            for (int offset = WarpSize/2; offset > 0; offset /= 2) 
                val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
            return val;
        }


        template <class T1, class T2, class T3,
                  std::size_t BlockSize,
                  std::size_t KernelSize,
                  std::size_t StrideSize,
                  std::size_t InputSize,
                  std::size_t ChanInSize,
                  std::size_t ChanOutSize>
        __global__ void conv2d_shared2(const T1 x, const T2 k, const T3 y)
        {

            constexpr std::size_t n_iters_tilex = (InputSize + BlockSize - 1) / BlockSize;
            constexpr std::size_t n_iters_tilek = (KernelSize + BlockSize - 1) / BlockSize;

            std::size_t y_row = threadIdx.x / BlockSize;
            std::size_t y_col = threadIdx.x % BlockSize;
            std::size_t y_cin = threadIdx.y / ChanOutSize;
            std::size_t y_cout = threadIdx.y % ChanOutSize;
            
            std::size_t y_i1 = blockIdx.x;
            std::size_t y_i4 = blockIdx.y * ChanOutSize + y_cout;
            

            dbl_t val = 0;

            std::size_t total_chans = k.d1();

            //store in shared memory
            //__shared__ dbl_t tile_k[KernelSize][KernelSize][ChanInSize];
            //__shared__ dbl_t tile_x[InputSize][InputSize][ChanInSize];

            __shared__ dbl_t tile_k[ChanOutSize][ChanInSize][KernelSize][KernelSize];
            __shared__ dbl_t tile_x[ChanInSize][InputSize][InputSize];
            
            for (std::size_t chan = 0; chan < total_chans; chan += ChanInSize)
            {


                for (std::size_t i = 0; i < n_iters_tilek; ++i)
                    for (std::size_t j = 0; j < n_iters_tilek; ++j)
                    {
                        std::size_t id_x = y_row + i * BlockSize;
                        std::size_t id_y = y_col + j * BlockSize;
                        if (id_x < KernelSize && id_y < KernelSize)
                        {
                            dbl_t vk = t_get(k, chan + y_cin, y_i4, id_x, id_y);
                            //tile_k[id_x][id_y][y_cin] = vk;
                            tile_k[y_cout][y_cin][id_x][id_y] = vk;
                        }
                    }


                for (std::size_t i = 0; i < n_iters_tilex; ++i)
                    for (std::size_t j = 0; j < n_iters_tilex; ++j)
                    {
                        std::size_t id_x = y_row + i * BlockSize;
                        std::size_t id_y = y_col + j * BlockSize;
                        if (id_x < InputSize && id_y < InputSize)
                        {
                            dbl_t vx = t_get(x, y_i1, chan + y_cin, id_x, id_y);
                            //tile_x[id_x][id_y][y_cin] = vx;
                            tile_x[y_cin][id_x][id_y] = vx;
                        }
                    }

                __syncthreads();
                    
                for (std::size_t i1 = 0; i1 < KernelSize; ++i1)
                    for (std::size_t i2 = 0; i2 < KernelSize; ++i2)
                    {
                        //dbl_t vx = tile_x[y_row*StrideSize + i1][y_col*StrideSize + i2][y_cin];
                        //dbl_t vk = tile_k[i1][i2][y_cin];

                        dbl_t vx = tile_x[y_cin][y_row*StrideSize + i1][y_col*StrideSize + i2];
                        dbl_t vk = tile_k[y_cout][y_cin][i1][i2];
                        
                        val += vx * vk;
                        
                    }

                __syncthreads();
            }

            tile_k[y_cout][y_cin][y_row][y_col] = val;
            __syncthreads();


            if (y_cin == 0)
            {

                dbl_t rval = 0;
                for (std::size_t i = 0; i < ChanInSize; ++i)
                    rval += tile_k[y_cout][i][y_row][y_col];
                
                dbl_t* ptr = y(y_i1, y_row, y_col, y_i4);
                *ptr = rval;
            }
        
        }
        

         
        void conv2d_fwd_shared2(const dbl_t* x, const dbl_t* k, dbl_t* y,
                                std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                                std::size_t pad_top, std::size_t pad_left,
                                std::size_t pad_bot, std::size_t pad_right,
                                std::size_t hk, std::size_t wk, std::size_t ck,
                                std::size_t hy, std::size_t wy,
                                std::size_t sh, std::size_t sw)
        {

            dbl_t* order_x;
            std::size_t full_x_len = nx * (hx + pad_top + pad_bot)
                * (wx + pad_left + pad_right) * cx;

            dbl_t* tmp = new dbl_t[full_x_len];
            for (std::size_t i = 0; i < full_x_len; ++i)
                tmp[i] = 0;
            
            cudaMalloc(&order_x, full_x_len * sizeof(dbl_t));
            cudaMemcpy(order_x, tmp, full_x_len * sizeof(dbl_t), cudaMemcpyHostToDevice);


            Tensor4<const dbl_t*> old_tx(x, nx, hx, wx, cx);
            Tensor4<dbl_t*> new_tx(order_x, nx, cx, hx + pad_top + pad_bot,
                                   wx + pad_left + pad_right);

            std::size_t len = old_tx.size();
            std::size_t block_size2 = 1024;
            std::size_t nb_blocks = (len + block_size2 - 1) / block_size2;
            prepare_input<<<nb_blocks, block_size2>>>(old_tx, new_tx, pad_top, pad_left);


            
            Tensor4<const dbl_t*> old_tk(k, hk, wk, cx, ck);
            dbl_t* order_k;
            cudaMalloc(&order_k, old_tk.size() * sizeof(dbl_t));
            //Tensor4<dbl_t*> new_tk(order_k, hk, wk, cx, ck);
            Tensor4<dbl_t*> new_tk(order_k, cx, ck, hk, wk);
            len = old_tk.size();
            nb_blocks = (len + block_size2 - 1) / block_size2;
            prepare_kernel<<<nb_blocks, block_size2>>>(old_tk, new_tk);


            Tensor4<dbl_t*> ty(y, nx, hy, wy, ck);
            
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);


            constexpr std::size_t kernel_size = 5;
            constexpr std::size_t stride_size = 2;

            using T1 = decltype(new_tx);
            using T2 = decltype(new_tk);
            using T3 = decltype(ty);


            int ntimes = 5;
            for (int i = 0; i < ntimes; ++i)
            {

                constexpr std::size_t block_size = 4;
                constexpr std::size_t input_size = 11;
                constexpr std::size_t chan_in_size = 32;
                constexpr std::size_t chan_out_size = 1;

                dim3 blocks_per_grid(ty.d1(), ty.d4() / chan_out_size);
                dim3 threads_per_block(block_size * block_size, chan_in_size * chan_out_size);

                std::size_t n_blocks = blocks_per_grid.x * blocks_per_grid.y;
                std::size_t n_threads = threads_per_block.x * threads_per_block.y;

                //std::cout << "n_blocks = " << n_blocks << std::endl;
                //std::cout << "n_threads = " << n_threads << std::endl;
                
                conv2d_shared2<T1, T2, T3,
                               block_size,
                               kernel_size,
                               stride_size,
                               input_size,
                               chan_in_size,
                               chan_out_size>
                    <<<blocks_per_grid, threads_per_block>>>(new_tx, new_tk, ty);

                /*
                if (hy == 32)
                {
                    constexpr std::size_t block_size = 32;
                    constexpr std::size_t input_size = 67;
                    constexpr std::size_t chan_out_size = 1;
                    dim3 blocks_per_grid(ty.d1(), ty.d4() / chan_out_size);
                    dim3 threads_per_block(block_size * block_size, chan_out_size);
                    
                    conv2d_shared2<T1, T2, T3,
                                   block_size,
                                   kernel_size,
                                   stride_size,
                                   input_size,
                                   chan_out_size>
                        <<<blocks_per_grid, threads_per_block>>>(new_tx, new_tk, ty);
                }

                else if (hy == 16)
                {
                    constexpr std::size_t block_size = 16;
                    constexpr std::size_t input_size = 35;
                    constexpr std::size_t chan_out_size = 1;
                    dim3 blocks_per_grid(ty.d1(), ty.d4() / chan_out_size);
                    dim3 threads_per_block(block_size * block_size, chan_out_size);
                    
                    conv2d_shared2<T1, T2, T3,
                                   block_size,
                                   kernel_size,
                                   stride_size,
                                   input_size,
                                   chan_out_size>
                        <<<blocks_per_grid, threads_per_block>>>(new_tx, new_tk, ty);
                }

                else if (hy == 8)
                {
                    constexpr std::size_t block_size = 8;
                    constexpr std::size_t input_size = 19;
                    constexpr std::size_t chan_out_size = 1;
                    dim3 blocks_per_grid(ty.d1(), ty.d4() / chan_out_size);
                    dim3 threads_per_block(block_size * block_size, chan_out_size);
                    
                    conv2d_shared2<T1, T2, T3,
                                   block_size,
                                   kernel_size,
                                   stride_size,
                                   input_size,
                                   chan_out_size>
                        <<<blocks_per_grid, threads_per_block>>>(new_tx, new_tk, ty);
                }

                else if (hy == 4)
                {
                    constexpr std::size_t block_size = 4;
                    constexpr std::size_t input_size = 11;
                    constexpr std::size_t chan_out_size = 64;//8
                    dim3 blocks_per_grid(ty.d1(), ty.d4() / chan_out_size);
                    dim3 threads_per_block(block_size * block_size, chan_out_size);
                    
                    conv2d_shared2<T1, T2, T3,
                                   block_size,
                                   kernel_size,
                                   stride_size,
                                   input_size,
                                   chan_out_size>
                        <<<blocks_per_grid, threads_per_block>>>(new_tx, new_tk, ty);
                }
                */

            }

            
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float time;
            cudaEventElapsedTime(&time, start, stop);

            time /= ntimes;
            std::ofstream fos("time.log", std::ios::app);
            fos << "time (fwd_shared2) = " << time << "ms\n" << std::endl;
        }
        
    }
    
}
