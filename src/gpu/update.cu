#include "update.hh"
#include "../runtime/node.hh"
#include <math_functions.h>

namespace gpu
{

    namespace
    {

        static constexpr std::size_t BLOCK_SIZE = 512;
        
        __global__
        void vect_update(const dbl_t* dv, dbl_t* out, const dbl_t* coeff, std::size_t len)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t i = index; i < len; i += stride)
                out[i] += *coeff * dv[i];
        }

        __global__
        void moment_update(const dbl_t* dv, dbl_t* out,
                           dbl_t a, dbl_t b,  std::size_t len)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t i = index; i < len; i += stride)
                out[i] = a * out[i] + b * dv[i];
        }

        __global__
        void moment_update2(const dbl_t* dv, dbl_t* out,
                            dbl_t a, dbl_t b,  std::size_t len)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t i = index; i < len; i += stride)
                out[i] = a * out[i] + b * dv[i] * dv[i];
        }

        __global__
        void adam_update(const dbl_t* m, const dbl_t* v, dbl_t* out,
                         dbl_t lrt, dbl_t eps, std::size_t len)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t i = index; i < len; i += stride)
                out[i] = out[i] - lrt * m[i] / (sqrt(v[i]) + eps);
        }
        
    }
    
    void kernel_update(rt::Node* node)
    {
        std::size_t len = node->len1;
        std::size_t nb_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vect_update<<<nb_blocks, BLOCK_SIZE>>>(node->in1, node->out1, node->in2, len);
    }

    void kernel_moment_update(rt::Node* node)
    {
        std::size_t len = node->len1;
        std::size_t nb_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        moment_update<<<nb_blocks, BLOCK_SIZE>>>(node->in1, node->out1,
                                                 node->cons1, node->cons2, len);
    }
    
    void kernel_moment_update2(rt::Node* node)
    {
        std::size_t len = node->len1;
        std::size_t nb_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        moment_update2<<<nb_blocks, BLOCK_SIZE>>>(node->in1, node->out1,
                                                  node->cons1, node->cons2, len);
    }

    void kernel_adam_update(rt::Node* node)
    {
        dbl_t* t = node->out2;
        dbl_t lr = node->cons1;
        dbl_t beta1 = node->cons2;
        dbl_t beta2 = node->cons3;
        dbl_t eps = node->cons4;
        ++*t;

        dbl_t lrt = lr * std::sqrt(1 - std::pow(beta2, *t))
            / (1 - std::pow(beta1, *t));

        std::size_t len = node->len1;
        std::size_t nb_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        adam_update<<<nb_blocks, BLOCK_SIZE>>>(node->in1, node->in2, node->out1, lrt, eps, len);
    }
   
}
