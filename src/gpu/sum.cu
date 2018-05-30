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

    }

    void kernel_mse(rt::Node* node)
    {
        std::size_t len = node->len1 * node->len2;
        mse<<<1, BLOCK_SIZE>>>(node->in1, node->in2, node->out1, len);
    }

}
