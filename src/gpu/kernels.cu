#include "kernels.hh"
#include "matmul.hh"
#include "sigmoid.hh"
#include "softmax.hh"
#include "sum.hh"

namespace gpu
{


    /**
     * blockDim: number of threads in a block
     * gridDim: number of blocks in a grid
     * blockIdx: current block index in the grid
     * threadIdx: current thread index in the block
     *
     * call<NB_BLOCKS, NB_THREADS_PER_BLOCK>
     */

    kernel_f kernels_list[512] = {
        kernel_mat_mat_mul,
        kernel_mat_rvect_add,
        kernel_sigmoid,
        kernel_mse,
        kernel_softmax,
        nullptr,//kernel_log_softmax,
        nullptr,//kernel_softmax_cross_entropy,
        nullptr,//kernel_conv2d,
        nullptr,//kernel_relu,
        nullptr,//kernel_relu_leaky,
        nullptr,//kernel_tanh,
        nullptr,//kernel_mse_grad,
        nullptr,//kernel_sigmoid_grad,
        nullptr,//kernel_mat_mul_add,
        nullptr,//kernel_tmat_mat_mul,
        nullptr,//kernel_mat_tmat_mul,
        nullptr,//kernel_mat_sum_rows,
        nullptr,//kernel_mat_sum_cols,
        nullptr,//kernel_softmax_cross_entropy_grad,
        nullptr,//kernel_relu_grad,
        nullptr,//kernel_conv2d_bias_add,
        nullptr,//kernel_update,
        nullptr,//kernel_sigmoid_cross_entropy,
        nullptr,//kernel_sigmoid_cross_entropy_grad,
        nullptr,//kernel_conv2d_input_grad,
        nullptr,//kernel_conv2d_kernel_grad,
        nullptr,//kernel_argmax_acc,
        nullptr,//kernel_moment_update,
        nullptr,//kernel_moment_update2,
        nullptr,//kernel_adam_update,
        nullptr,//kernel_leaky_relu_grad,
        nullptr,//kernel_conv2d_bias_add_grad,
        nullptr,//kernel_tanh_grad,
        nullptr,//kernel_conv2d_transpose,
        nullptr,//kernel_conv2d_transpose_input_grad,
        nullptr,//kernel_conv2d_transpose_kernel_grad
    };

}
