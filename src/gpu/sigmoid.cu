#include "kernels.hh"
#include <math_functions.h>
#include "../runtime/node.hh"



namespace gpu
{

    namespace
    {

        __device__
        dbl_t sigmoid(dbl_t x)
        {
            return 1.0 / (1.0 + exp(-x));
        }

        __global__
        void vect_sigmoid(const dbl_t* x, dbl_t* y, std::size_t len)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t i = index; i < len; i += stride)
                y[i] = sigmoid(x[i]);
        }

        __global__
        void vect_tanh(const dbl_t* x, dbl_t* y, std::size_t len)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t i = index; i < len; i += stride)
                y[i] = tanh(x[i]);
        }

    }
        

    void kernel_sigmoid(rt::Node* node)
    {
        std::size_t len = node->len1;
        std::size_t block_size = 256;
        std::size_t nb_blocks = (len + block_size - 1)/ block_size;

        vect_sigmoid<<<nb_blocks, block_size>>>(node->in1, node->out1, len);
    }

    void kernel_tanh(rt::Node* node)
    {
        std::size_t len = node->len1;
        std::size_t block_size = 256;
        std::size_t nb_blocks = (len + block_size - 1)/ block_size;

        vect_tanh<<<nb_blocks, block_size>>>(node->in1, node->out1, len);
    }

}
