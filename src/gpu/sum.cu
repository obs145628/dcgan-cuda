#include "sum.hh"
#include "../runtime/node.hh"



namespace gpu
{

    namespace
    {

        constexpr std::size_t BLOCK_SIZE = 512;

        __global__
        void mse(const dbl_t* a, const dbl_t* b, dbl_t* out,
                 std::size_t len)
        {
            __shared__ float partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto i = threadIdx.x;
            std::size_t step = BLOCK_SIZE;

            float init = 0;
            for (std::size_t j = i; j < len; j += step)
                init += (a[j] - b[j]) * (a[j] - b[j]);
        
            partial[i] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (i < s)
                    partial[i] += partial[i + s];

                __syncthreads();
            }

            //if not volatile, must use __synctthreads again, why ?
            volatile float* vpartial = partial;
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
                *out = partial[0] / len;
        }

        __global__
        void mse_grad(const dbl_t* a, const dbl_t* b, dbl_t* out, std::size_t len)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;
            dbl_t coeff = dbl_t(2) / len;

            for (std::size_t i = index; i < len; i += stride)
                out[i] = coeff * (a[i] - b[i]);
        }


        __global__
        void mat_sum_rows(const dbl_t* x, dbl_t* y,
                          std::size_t rows, std::size_t cols)
        {
            __shared__ dbl_t partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto row = blockIdx.x;
            auto col = threadIdx.x;
            std::size_t step = BLOCK_SIZE;

            dbl_t init = 0;
            for (std::size_t i = col; i < cols; i += step)
                init += x[row * cols + i];
        
            partial[col] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (col < s)
                    partial[col] += partial[col + s];

                __syncthreads();
            }

            //if not volatile, must use __synctthreads again, why ?
            volatile dbl_t* vpartial = partial;
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
                y[row] = partial[0];
        }


        __global__
        void mat_sum_cols(const dbl_t* x, dbl_t* y,
                          std::size_t rows, std::size_t cols)
        {
            __shared__ dbl_t partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto col = blockIdx.x;
            auto row = threadIdx.x;
            std::size_t step = BLOCK_SIZE;
            
            dbl_t init = 0;
            for (std::size_t i = row; i < rows; i += step)
                init += x[i * cols + col];
        
            partial[row] = init;
            __syncthreads();

            for (std::size_t s = BLOCK_SIZE / 2; s > 32; s >>= 1)
            {
                if (row < s)
                    partial[row] += partial[row + s];

                __syncthreads();
            }

            //if not volatile, must use __synctthreads again, why ?
            volatile dbl_t* vpartial = partial;
            if (row < 32)
            {
                vpartial[row] += vpartial[row + 32];
                vpartial[row] += vpartial[row + 16];
                vpartial[row] += vpartial[row + 8];
                vpartial[row] += vpartial[row + 4];
                vpartial[row] += vpartial[row + 2];
                vpartial[row] += vpartial[row + 1];
            }
            
            if (row == 0)
                y[col] = partial[0];
        }

        __device__
        std::size_t argmax(const dbl_t* begin, const dbl_t* end)
        {
            const dbl_t* res = begin;
            for (const dbl_t* it = begin; it != end; ++it)
                if (*it > *res)
                    res = it;
            return res - begin;
        }

        __global__
        void argmax_acc(const dbl_t* a, const dbl_t* b, dbl_t* out,
                        std::size_t rows, std::size_t cols)
        {
            __shared__ dbl_t partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto i = threadIdx.x;
            std::size_t step = BLOCK_SIZE;

            dbl_t init = 0;
            for (std::size_t j = i; j < rows; j += step)
                init += argmax(a + j * cols, a + (j + 1) * cols)  
                    == argmax(b + j * cols, b + (j + 1) * cols);
                    
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
                *out = partial[0];
        }

    }

    void kernel_mse(rt::Node* node)
    {
        std::size_t len = node->len1 * node->len2;
        mse<<<1, BLOCK_SIZE>>>(node->in1, node->in2, node->out1, len);
    }

    void kernel_mse_grad(rt::Node* node)
    {
        std::size_t len = node->len1;
        std::size_t nb_blocks = (len + BLOCK_SIZE - 1)/ BLOCK_SIZE;
        mse_grad<<<nb_blocks, BLOCK_SIZE>>>(node->in2, node->in1, node->out1, node->len1);
    }

    void kernel_mat_sum_rows(rt::Node* node)
    {
        std::size_t rows = node->len1;
        std::size_t cols = node->len2;
        mat_sum_rows<<<rows, BLOCK_SIZE>>>(node->in1, node->out1, rows, cols);
    }

    void kernel_mat_sum_cols(rt::Node* node)
    {
        std::size_t rows = node->len1;
        std::size_t cols = node->len2;
        mat_sum_cols<<<cols, BLOCK_SIZE>>>(node->in1, node->out1, rows, cols);
    }

    void kernel_argmax_acc(rt::Node* node)
    {
        argmax_acc<<<1, BLOCK_SIZE>>>(node->in1, node->in2, node->out1,
                                      node->len1, node->len2);
    }

}
