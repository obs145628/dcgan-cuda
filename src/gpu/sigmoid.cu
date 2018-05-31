#include "kernels.hh"
#include <math_functions.h>
#include "../runtime/node.hh"



namespace gpu
{

    namespace
    {

        static constexpr std::size_t BLOCK_SIZE = 512;

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

        __device__
        dbl_t cross_sigmoid(dbl_t x, dbl_t y)
        {
            return x >= 0 ?
                x - x * y + log(1 + exp(-x)) :
                - x * y + log(1 + exp(x));
        }

        inline dbl_t sigmoid_cross_entropy(const dbl_t* y, const dbl_t* x,
                                           std::size_t n)
        {
            dbl_t res = 0;
            for (std::size_t i = 0; i < n; ++i)
            {
                if (x[i] >= 0)
                    res += x[i] - x[i] * y[i] + std::log(1 + std::exp(-x[i]));
                else
                    res += - x[i] * y[i] + std::log(1 + std::exp(x[i]));
            }
            return res / n;
        }

        __global__
        void sigmoid_cross_entropy(const dbl_t* y, const dbl_t* x, dbl_t* out,
                                   std::size_t len)
        {
            __shared__ dbl_t partial[2 * BLOCK_SIZE];

            //load all elements of the array in shared memory
            auto i = threadIdx.x;
            std::size_t step = BLOCK_SIZE;

            dbl_t init = 0;
            for (std::size_t j = i; j < len; j += step)
                init += cross_sigmoid(x[j], y[j]);
        
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
                *out = partial[0] / len;
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

    void kernel_sigmoid_cross_entropy(rt::Node* node)
    {
        std::size_t len = node->len1;
        sigmoid_cross_entropy<<<1, BLOCK_SIZE>>>(node->in1, node->in2, node->out1, len);
    }

}
