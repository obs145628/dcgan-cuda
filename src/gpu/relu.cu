#include "kernels.hh"
#include <math_functions.h>
#include "../runtime/node.hh"



namespace gpu
{

    namespace
    {

        __device__
        dbl_t relu(dbl_t x)
        {
            return max(dbl_t(0), x);
        }


        __device__
        dbl_t relu_leaky(dbl_t x, dbl_t alpha)
        {
            return x < 0 ? alpha * x : x;
        }

        __global__
        void vect_relu(const dbl_t* x, dbl_t* y, std::size_t len)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t i = index; i < len; i += stride)
                y[i] = relu(x[i]);
        }

        __global__
        void vect_relu_grad(const dbl_t* z, const dbl_t* dout, dbl_t* out, std::size_t len)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t i = index; i < len; i += stride)
                out[i] = z[i] > 0 ? dout[i] : 0;
        }

        __global__
        void vect_relu_leaky(const dbl_t* x, dbl_t* y, dbl_t alpha, std::size_t len)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t i = index; i < len; i += stride)
                y[i] = relu_leaky(x[i], alpha);
        }

        __global__
        void vect_relu_leaky_grad(const dbl_t* z, const dbl_t* dout, dbl_t* out,
                                  dbl_t alpha, std::size_t len)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t i = index; i < len; i += stride)
                out[i] = z[i] < 0 ? alpha * dout[i] : dout[i];
        }
    }
        

    void kernel_relu(rt::Node* node)
    {
        std::size_t len = node->len1;
        std::size_t block_size = 256;
        std::size_t nb_blocks = (len + block_size - 1)/ block_size;

        vect_relu<<<nb_blocks, block_size>>>(node->in1, node->out1, len);
    }

    void kernel_relu_grad(rt::Node* node)
    {
        std::size_t len = node->len1;
        std::size_t block_size = 256;
        std::size_t nb_blocks = (len + block_size - 1)/ block_size;

        vect_relu_grad<<<nb_blocks, block_size>>>(node->in1, node->in2, node->out1, len);
    }

    void kernel_relu_leaky(rt::Node* node)
    {
        std::size_t len = node->len1;
        std::size_t block_size = 256;
        std::size_t nb_blocks = (len + block_size - 1)/ block_size;

        vect_relu_leaky<<<nb_blocks, block_size>>>(node->in1, node->out1,
                                                   node->alpha_leaky, len);
    }

    void kernel_leaky_relu_grad(rt::Node* node)
    {
        std::size_t len = node->len1;
        std::size_t block_size = 256;
        std::size_t nb_blocks = (len + block_size - 1)/ block_size;

        vect_relu_leaky_grad<<<nb_blocks, block_size>>>(node->in1, node->in2, node->out1,
                                                        node->cons1, len);
    }

}
