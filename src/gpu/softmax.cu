#include "softmax.hh"
#include "../runtime/node.hh"



namespace gpu
{

    namespace
    {
        constexpr std::size_t BLOCK_SIZE = 512;

        __global__
        void softmax1(const dbl_t* x, dbl_t* y,
                      std::size_t rows, std::size_t cols) //8ms
        {
            __shared__ dbl_t partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto row = blockIdx.x;
            auto col = threadIdx.x;
            std::size_t step = BLOCK_SIZE;

            dbl_t init = 1e-30;
            for (std::size_t i = col; i < cols; i += step)
                init = max(x[row * cols + i], init);
        
            partial[col] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (col < s)
                    partial[col] = max(partial[col], partial[col + s]);

                __syncthreads();
            }

            //if not volatile, must use __synctthreads again, why ?
            volatile dbl_t* vpartial = partial;
            if (col < 32)
            {
                vpartial[col] = max(vpartial[col], vpartial[col + 32]);
                vpartial[col] = max(vpartial[col], vpartial[col + 16]);
                vpartial[col] = max(vpartial[col], vpartial[col + 8]);
                vpartial[col] = max(vpartial[col], vpartial[col + 4]);
                vpartial[col] = max(vpartial[col],vpartial[col + 2]);
                vpartial[col] = max(vpartial[col], vpartial[col + 1]);
            }
            
            __syncthreads();
            
            for (std::size_t i = col; i < cols; i += step)
                y[row * cols + i] = exp(x[row * cols + i] - partial[0]);

            __syncthreads();

            init = 0;
            for (std::size_t i = col; i < cols; i += step)
                init += y[row * cols + i];
        
            partial[col] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (col < s)
                    partial[col] += partial[col + s];

                __syncthreads();
            }

            if (col < 32)
            {
                vpartial[col] += vpartial[col + 32];
                vpartial[col] += vpartial[col + 16];
                vpartial[col] += vpartial[col + 8];
                vpartial[col] += vpartial[col + 4];
                vpartial[col] += vpartial[col + 2];
                vpartial[col] += vpartial[col + 1];
            }


            __syncthreads();
            
            for (std::size_t i = col; i < cols; i += step)
                y[row * cols + i] /= partial[0];
        }

        __global__
        void log_softmax1(const dbl_t* x, dbl_t* y,
                      std::size_t rows, std::size_t cols) //8ms
        {
            __shared__ dbl_t partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto row = blockIdx.x;
            auto col = threadIdx.x;
            std::size_t step = BLOCK_SIZE;

            dbl_t init = 1e-30;
            for (std::size_t i = col; i < cols; i += step)
                init = max(x[row * cols + i], init);
        
            partial[col] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (col < s)
                    partial[col] = max(partial[col], partial[col + s]);

                __syncthreads();
            }

            //if not volatile, must use __synctthreads again, why ?
            volatile dbl_t* vpartial = partial;
            if (col < 32)
            {
                vpartial[col] = max(vpartial[col], vpartial[col + 32]);
                vpartial[col] = max(vpartial[col], vpartial[col + 16]);
                vpartial[col] = max(vpartial[col], vpartial[col + 8]);
                vpartial[col] = max(vpartial[col], vpartial[col + 4]);
                vpartial[col] = max(vpartial[col],vpartial[col + 2]);
                vpartial[col] = max(vpartial[col], vpartial[col + 1]);
            }
            
            __syncthreads();

            dbl_t max_x = partial[0];

            init = 0;
            for (std::size_t i = col; i < cols; i += step)
                init += exp(x[row * cols + i] - max_x);
        
            partial[col] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (col < s)
                    partial[col] += partial[col + s];

                __syncthreads();
            }

            if (col < 32)
            {
                vpartial[col] += vpartial[col + 32];
                vpartial[col] += vpartial[col + 16];
                vpartial[col] += vpartial[col + 8];
                vpartial[col] += vpartial[col + 4];
                vpartial[col] += vpartial[col + 2];
                vpartial[col] += vpartial[col + 1];
            }


            __syncthreads();

            dbl_t logsum = max_x + std::log(partial[0]);
            
            for (std::size_t i = col; i < cols; i += step)
                y[row * cols + i] = x[row * cols + i] - logsum;
        }

        __global__
        void softmax_lcost1(const dbl_t* y, const dbl_t* x, dbl_t* out,
                            std::size_t rows, std::size_t cols)
        {
            __shared__ dbl_t partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto row = blockIdx.x;
            auto col = threadIdx.x;
            std::size_t step = BLOCK_SIZE;

            dbl_t init = 1e-30;
            for (std::size_t i = col; i < cols; i += step)
                init = max(x[row * cols + i], init);
        
            partial[col] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (col < s)
                    partial[col] = max(partial[col], partial[col + s]);

                __syncthreads();
            }

            //if not volatile, must use __synctthreads again, why ?
            volatile dbl_t* vpartial = partial;
            if (col < 32)
            {
                vpartial[col] = max(vpartial[col], vpartial[col + 32]);
                vpartial[col] = max(vpartial[col], vpartial[col + 16]);
                vpartial[col] = max(vpartial[col], vpartial[col + 8]);
                vpartial[col] = max(vpartial[col], vpartial[col + 4]);
                vpartial[col] = max(vpartial[col],vpartial[col + 2]);
                vpartial[col] = max(vpartial[col], vpartial[col + 1]);
            }
            
            __syncthreads();

            dbl_t max_x = partial[0];

            init = 0;
            for (std::size_t i = col; i < cols; i += step)
                init += exp(x[row * cols + i] - max_x);
        
            partial[col] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (col < s)
                    partial[col] += partial[col + s];

                __syncthreads();
            }

            if (col < 32)
            {
                vpartial[col] += vpartial[col + 32];
                vpartial[col] += vpartial[col + 16];
                vpartial[col] += vpartial[col + 8];
                vpartial[col] += vpartial[col + 4];
                vpartial[col] += vpartial[col + 2];
                vpartial[col] += vpartial[col + 1];
            }


            __syncthreads();

            dbl_t logsum = max_x + std::log(partial[0]);

            init = 0;
            for (std::size_t i = col; i < cols; i += step)
                init += y[row * cols + i] * (x[row * cols + i] - logsum);
        
            partial[col] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (col < s)
                    partial[col] += partial[col + s];

                __syncthreads();
            }

            if (col < 32)
            {
                vpartial[col] += vpartial[col + 32];
                vpartial[col] += vpartial[col + 16];
                vpartial[col] += vpartial[col + 8];
                vpartial[col] += vpartial[col + 4];
                vpartial[col] += vpartial[col + 2];
                vpartial[col] += vpartial[col + 1];
            }
            
            if (col == 0)
                out[row] = partial[0];
        }

        __global__
        void softmax_reduce(const dbl_t* x, dbl_t* y, std::size_t len) //8ms
        {
            __shared__ dbl_t partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto i = threadIdx.x;
            std::size_t step = BLOCK_SIZE;

            dbl_t init = 0;
            for (std::size_t j = i; j < len; j += step)
                init += x[j];
        
            partial[i] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (i < s)
                    partial[i] += partial[i + s];

                __syncthreads();
            }

            //if not volatile, must use __synctthreads again, why ?
            volatile dbl_t* vpartial = partial;
            if (i < 32)
            {
                vpartial[i] += vpartial[i + 32];
                vpartial[i] += vpartial[i + 16];
                vpartial[i] += vpartial[i + 8];
                vpartial[i] += vpartial[i + 4];
                vpartial[i] += vpartial[i + 2];
                vpartial[i] += vpartial[i + 1];
            }


            if (i == 0)
                y[0] = - partial[0] / len;
        }

        __global__
        void softmax_cross_entropy_grad(const dbl_t* y, dbl_t* out,
                                        std::size_t rows, std::size_t cols)
        {
            std::size_t len = rows * cols;
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t i = index; i < len; i += stride)
                out[i] = (out[i] - y[i]) / rows;
        }
    }

    void kernel_softmax(rt::Node* node)
    {
        std::size_t rows = node->len1;
        std::size_t cols = node->len2;
        softmax1<<<rows, BLOCK_SIZE>>>(node->in1, node->out1, rows, cols);
    }

    void kernel_log_softmax(rt::Node* node)
    {
        std::size_t rows = node->len1;
        std::size_t cols = node->len2;
        log_softmax1<<<rows, BLOCK_SIZE>>>(node->in1, node->out1, rows, cols);
    }

    void kernel_softmax_cross_entropy(rt::Node* node)
    {
        std::size_t rows = node->len1;
        std::size_t cols = node->len2;
        const dbl_t* y = node->in1;
        const dbl_t* x = node->in2;
        dbl_t* out = node->out1;

        dbl_t* tmp;
        cudaMalloc(&tmp, rows * sizeof(dbl_t));
        softmax_lcost1<<<rows, BLOCK_SIZE>>>(y, x, tmp, rows, cols);
        cudaDeviceSynchronize();
        softmax_reduce<<<1, BLOCK_SIZE>>>(tmp, out, rows);
        cudaDeviceSynchronize();
        cudaFree(tmp);
    }

    void kernel_softmax_cross_entropy_grad(rt::Node* node)
    {
        const dbl_t* y = node->in1;
        const dbl_t* logits = node->in2;
        dbl_t* out = node->out1;
        std::size_t rows = node->len1;
        std::size_t cols = node->len2;
        
        softmax1<<<rows, BLOCK_SIZE>>>(logits, out, rows, cols);
        cudaDeviceSynchronize();

        std::size_t len = rows * cols;
        std::size_t nb_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        softmax_cross_entropy_grad<<<nb_blocks, BLOCK_SIZE>>>(y, out, rows, cols);
    }

}
