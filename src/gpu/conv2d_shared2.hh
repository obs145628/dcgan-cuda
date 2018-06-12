#pragma once

#include <fstream>

namespace gpu
{

    namespace
    {

        /*
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
            dbl_t* optr = out(i1, i2 + pad_top, i3 + pad_left, i4);
            //dbl_t* optr = out(i1, i4, i2 + pad_top, i3 + pad_left);
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
            //dbl_t* optr = out(i4, i3, i1, i2);
            dbl_t* optr = out(i4, i1, i2, i3);
            *optr = *iptr;
        }
        */




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


        /**
         * Best version for now
         * Warning: broken because i changed order of kernel k
         *
        template <class T1, class T2, class T3,
                  std::size_t BlockSize,
                  std::size_t KernelSize,
                  std::size_t StrideSize,
                  std::size_t InputSize,
                  std::size_t ChanInSize>
        __global__ void conv2d_shared2(const T1 x, const T2 k, const T3 y)
        {

            constexpr std::size_t n_iters_tilex = (InputSize + BlockSize - 1) / BlockSize;
            constexpr std::size_t n_iters_tilek = (KernelSize + BlockSize - 1) / BlockSize;

            std::size_t y_row = threadIdx.x / BlockSize;
            std::size_t y_col = threadIdx.x % BlockSize;
            std::size_t y_cin = threadIdx.y;
            
            std::size_t y_i1 = blockIdx.x;
            std::size_t y_i4 = blockIdx.y;

            dbl_t val = 0;

            std::size_t total_chans = k.d1();

            __shared__ dbl_t tile_k[ChanInSize][KernelSize][KernelSize];
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
                            tile_k[y_cin][id_x][id_y] = vk;
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
                            tile_x[y_cin][id_x][id_y] = vx;
                        }
                    }

                __syncthreads();
                    
                for (std::size_t i1 = 0; i1 < KernelSize; ++i1)
                    for (std::size_t i2 = 0; i2 < KernelSize; ++i2)
                    {
                        dbl_t vx = tile_x[y_cin][y_row*StrideSize + i1][y_col*StrideSize + i2];
                        dbl_t vk = tile_k[y_cin][i1][i2];   
                        val += vx * vk;
                        
                    }

                __syncthreads();
            }

            tile_k[y_cin][y_row][y_col] = val;
            __syncthreads();


            if (y_cin == 0)
            {

                dbl_t rval = 0;
                for (std::size_t i = 0; i < ChanInSize; ++i)
                    rval += tile_k[i][y_row][y_col];
                
                dbl_t* ptr = y(y_i1, y_row, y_col, y_i4);
                *ptr = rval;
            }
        
        }
        */


        template <std::size_t BlockSize,
                  std::size_t KernelSize,
                  std::size_t StrideSize,
                  std::size_t InputSize,
                  std::size_t D3Size,
                  std::size_t D4Size,
                  std::size_t ChanInSize>
        __global__ void conv2d_shared2(__restrict__ const dbl_t* x_ptr,
                                       __restrict__ const dbl_t* k_ptr,
                                       __restrict__ dbl_t* y_ptr)
        {
            
            constexpr std::size_t nb_threads = BlockSize * BlockSize * ChanInSize;
            constexpr std::size_t tile_x_size = ChanInSize * InputSize * InputSize;
            constexpr std::size_t tile_k_size = ChanInSize * KernelSize * KernelSize;
            //constexpr std::size_t nb_iters_tile_x = (tile_x_size + nb_threads - 1) / nb_threads;
            //constexpr std::size_t nb_iters_tile_k = (tile_k_size + nb_threads - 1) / nb_threads;
            constexpr std::size_t total_chans = D3Size;

            constexpr std::size_t n_iters_tilek = (KernelSize * KernelSize
                                                   + BlockSize * BlockSize - 1)
                / (BlockSize * BlockSize);


            constexpr std::size_t PadTop = 1;
            constexpr std::size_t PadLeft = 1;
            constexpr std::size_t PadBottom = 2*PadTop;
            constexpr std::size_t PadRight = 2*PadTop;
            
            constexpr std::size_t true_input_size = InputSize - PadTop - PadBottom;

            constexpr std::size_t n_iters_tilex = (true_input_size + BlockSize - 1) / BlockSize;

            
            
            const std::size_t y_row = threadIdx.y / BlockSize;
            const std::size_t y_col = threadIdx.y % BlockSize;
            const std::size_t y_cin = threadIdx.x;
            
            const std::size_t y_i1 = blockIdx.x;
            const std::size_t y_i4 = blockIdx.y;

            const std::size_t tid = threadIdx.y * ChanInSize + threadIdx.x;
            
            
            dbl_t val = 0;

            __shared__ dbl_t tile_x[InputSize][InputSize][ChanInSize];
            dbl_t* tile_x_ptr = tile_x[0][0];
            __shared__ dbl_t tile_k[KernelSize][KernelSize][ChanInSize];
            dbl_t* tile_k_ptr = tile_k[0][0];

            //0 init tile_x

            std::size_t ttnn = (tile_x_size + nb_threads - 1) / nb_threads; 
            #pragma unroll
            for (std::size_t i = 0; i < ttnn; ++i)
            {
                std::size_t idx = tid + i * nb_threads;
                if (i + 1 < ttnn || idx < tile_x_size)
                    tile_x_ptr[idx] = 0;
            }
            __syncthreads();

            #pragma unroll
            for (std::size_t chan = 0; chan < total_chans; chan += ChanInSize)
            {

                const dbl_t* x_ptr_chan = x_ptr
                    + (y_i1) * (true_input_size * true_input_size * D3Size)
                    + (chan);

                #pragma unroll
                for (std::size_t i = 0; i < n_iters_tilex; ++i)
                    for (std::size_t j = 0; j < n_iters_tilex; ++j)
                    {
                        std::size_t id_x = y_row + PadTop + i * BlockSize;
                        std::size_t id_y = y_col + PadLeft + j * BlockSize;
                        
                        if (i + 1 < n_iters_tilex ||
                            (id_x < InputSize - PadBottom
                             && id_y < InputSize - PadRight))
                        {
                            const std::size_t true_id_x = id_x - PadTop;
                            const std::size_t true_id_y = id_y - PadLeft;
                            
                            dbl_t vx = x_ptr_chan[true_id_x * true_input_size * D3Size
                                                  + true_id_y * D3Size + y_cin];
                            tile_x[id_x][id_y][y_cin] = vx;
                        }
                    }

                const dbl_t* k_ptr_chan = k_ptr
                    + (y_i4) * (1)
                    + (chan) * (D4Size);
                

                #pragma unroll
                for (std::size_t i = 0; i < n_iters_tilek; ++i)
                {
                    std::size_t idx = i * BlockSize * BlockSize + threadIdx.y;
                    if (i + 1 < n_iters_tilek || idx < KernelSize * KernelSize)
                    {
                        dbl_t vk = k_ptr_chan[idx * D3Size * D4Size + y_cin * D4Size];
                        tile_k_ptr[idx * ChanInSize + y_cin] = vk;
                    }
                }

                __syncthreads();


                /*
                if (chan == 0 && tid == 0 && y_i4 == 0 && y_i1 == 0)
                {
                    for (std::size_t i = 0; i < 11; ++i)
                    {
                        for (std::size_t j = 0; j < 11; ++j)
                        {
                            printf("%4.6f ", tile_x[i][j][0]);
                        }
                        printf("\n");
                    }
                    printf("\n\n");
                }
                */

                // + opti: 32 threads lisent en coalescent dans la shared (no bank conflict)
                // difficile pour x a cause de la stride size
                // faisable pour k
                //seul difference: y_cin
                //wrap 0: y_cin = 0, wrap 1: y_cin = 1, ...
                //not-coallessing access: dbl_t vk = tile_k[i1][i2][y_cin];

                #pragma unroll
                for (std::size_t i1 = 0; i1 < KernelSize; ++i1)
                    #pragma unroll
                    for (std::size_t i2 = 0; i2 < KernelSize; ++i2)
                    {
                        //dbl_t vx = tile_x[y_cin][y_row*StrideSize + i1][y_col*StrideSize + i2];
                        dbl_t vx = tile_x[y_row*StrideSize + i1][y_col*StrideSize + i2][y_cin];

                        //dbl_t vk = tile_k[y_cin][i1][i2];
                        dbl_t vk = tile_k[i1][i2][y_cin];

                        val += vx * vk;
                    }

                __syncthreads();
            }

            val = warp_reduce_sum<ChanInSize>(val);

            //tile_k[y_row][y_col][y_cin] = val;
            __syncthreads();

            if (y_cin == 0)
            {

                dbl_t* ptr = y_ptr + y_i1 * BlockSize * BlockSize * D4Size
                    + y_row * BlockSize * D4Size
                    + y_col * D4Size + y_i4;
                *ptr = val;
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
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            constexpr std::size_t kernel_size = 5;
            constexpr std::size_t stride_size = 2;


            int ntimes = 1;
            for (int i = 0; i < ntimes; ++i)
            {


                if (hy == 32)
                {
                    constexpr std::size_t block_size = 32;
                    constexpr std::size_t input_size = 67;
                    constexpr std::size_t d3_size = 3;
                    constexpr std::size_t d4_size = 64;
                    constexpr std::size_t chan_in_size = 1;
                    

                    dim3 blocks_per_grid(nx, ck);
                    dim3 threads_per_block(chan_in_size, block_size * block_size);
                
                    conv2d_shared2<block_size,
                                   kernel_size,
                                   stride_size,
                                   input_size,
                                   d3_size,
                                   d4_size,
                                   chan_in_size>
                        <<<blocks_per_grid, threads_per_block>>>(x, k, y);
                }

                else if (hy == 16)
                {
                    constexpr std::size_t block_size = 16;
                    constexpr std::size_t input_size = 35;
                    constexpr std::size_t d3_size = 64;
                    constexpr std::size_t d4_size = 128;
                    constexpr std::size_t chan_in_size = 4;
                    

                    dim3 blocks_per_grid(nx, ck);
                    dim3 threads_per_block(chan_in_size, block_size * block_size);
                
                    conv2d_shared2<block_size,
                                   kernel_size,
                                   stride_size,
                                   input_size,
                                   d3_size,
                                   d4_size,
                                   chan_in_size>
                        <<<blocks_per_grid, threads_per_block>>>(x, k, y);
                }

                else if (hy == 8)
                {
                    constexpr std::size_t block_size = 8;
                    constexpr std::size_t input_size = 19;
                    constexpr std::size_t d3_size = 128;
                    constexpr std::size_t d4_size = 256;
                    constexpr std::size_t chan_in_size = 16;
                    

                    dim3 blocks_per_grid(nx, ck);
                    dim3 threads_per_block(chan_in_size, block_size * block_size);
                
                    conv2d_shared2<block_size,
                                   kernel_size,
                                   stride_size,
                                   input_size,
                                   d3_size,
                                   d4_size,
                                   chan_in_size>
                        <<<blocks_per_grid, threads_per_block>>>(x, k, y);
                }

                else if (hy == 4)
                {

                    constexpr std::size_t block_size = 4;
                    constexpr std::size_t input_size = 11;
                    constexpr std::size_t d3_size = 256;
                    constexpr std::size_t d4_size = 512;
                    constexpr std::size_t chan_in_size = 32;
                    

                    dim3 blocks_per_grid(nx, ck);
                    dim3 threads_per_block(chan_in_size, block_size * block_size);
                
                    conv2d_shared2<block_size,
                                   kernel_size,
                                   stride_size,
                                   input_size,
                                   d3_size,
                                   d4_size,
                                   chan_in_size>
                        <<<blocks_per_grid, threads_per_block>>>(x, k, y);
                }

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
            //std::ofstream fos("time.log", std::ios::app);
            //fos << "time (fwd_shared2) = " << time << "ms\n" << std::endl;
        }
        
    }
    
}
