#include "matmul.hh"
#include "../runtime/node.hh"



namespace gpu
{

    namespace
    {

        __global__
        void matmul(const dbl_t* a, const dbl_t* b, dbl_t* out,
                    std::size_t arows, std::size_t acols, std::size_t bcols)
        {
            std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t col = blockIdx.y * blockDim.y + threadIdx.y;

            if (row >= arows || col >= bcols)
                return;

            dbl_t x = 0;
            for (std::size_t i = 0; i < acols; ++i)
                x += a[row * acols + i] * b[i * bcols + col];
            out[row * bcols + col] = x;
        }

        __global__
        void mvrow_add(const dbl_t* a, const dbl_t* b, dbl_t* out,
                       std::size_t rows, std::size_t cols)
        {
            std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t col = blockIdx.y * blockDim.y + threadIdx.y;

            if (row >= rows || col >= cols)
                return;

            out[row * cols + col] = a[row * cols + col] + b[col];
        }

        __global__
        void matmul_add(const dbl_t* a, const dbl_t* b, const dbl_t* c, dbl_t* out,
                        std::size_t arows, std::size_t acols, std::size_t bcols)
        {
            std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t col = blockIdx.y * blockDim.y + threadIdx.y;

            if (row >= arows || col >= bcols)
                return;

            dbl_t x = 0;
            for (std::size_t i = 0; i < acols; ++i)
                x += a[row * acols + i] * b[i * bcols + col];
            out[row * bcols + col] = x + c[col];
        }
    }
        

    void kernel_mat_mat_mul(rt::Node* node)
    {
        std::size_t arows = node->len1;
        std::size_t acols = node->len2;
        std::size_t bcols = node->len3;

        std::size_t block_size = 32;
        dim3 threads_per_block (block_size, block_size);
        std::size_t nb_blocks_x = (arows + block_size - 1) / block_size;
        std::size_t nb_blocks_y = (bcols + block_size - 1) / block_size;
        dim3 blocks_per_grid (nb_blocks_x, nb_blocks_y);

        matmul<<<blocks_per_grid, threads_per_block>>>(node->in1, node->in2, node->out1,
                                                       arows, acols, bcols);
    }

    void kernel_mat_rvect_add(rt::Node* node)
    {
        std::size_t rows = node->len1;
        std::size_t cols = node->len2;

        std::size_t block_size = 32;
        dim3 threads_per_block (block_size, block_size);
        std::size_t nb_blocks_x = (rows + block_size - 1) / block_size;
        std::size_t nb_blocks_y = (cols + block_size - 1) / block_size;
        dim3 blocks_per_grid (nb_blocks_x, nb_blocks_y);

        mvrow_add<<<blocks_per_grid, threads_per_block>>>(node->in1, node->in2, node->out1,
                                                          rows, cols);
    }

    void kernel_mat_mul_add(rt::Node* node)
    {
        std::size_t arows = node->len1;
        std::size_t acols = node->len2;
        std::size_t bcols = node->len3;

        std::size_t block_size = 32;
        dim3 threads_per_block (block_size, block_size);
        std::size_t nb_blocks_x = (arows + block_size - 1) / block_size;
        std::size_t nb_blocks_y = (bcols + block_size - 1) / block_size;
        dim3 blocks_per_grid (nb_blocks_x, nb_blocks_y);

        matmul_add<<<blocks_per_grid, threads_per_block>>>(node->in1, node->in2, node->in3,
                                                           node->out1,
                                                           arows, acols, bcols);
    }

}
