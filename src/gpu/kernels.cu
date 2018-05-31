#include "kernels.hh"
#include "matmul.hh"
#include "sigmoid.hh"
#include "softmax.hh"
#include "relu.hh"
#include "sum.hh"
#include "update.hh"

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
        kernel_log_softmax,
        kernel_softmax_cross_entropy,
        nullptr,//kernel_conv2d,
        kernel_relu,
        kernel_relu_leaky,
        kernel_tanh,
        kernel_mse_grad,
        kernel_sigmoid_grad,
        kernel_mat_mul_add,
        kernel_tmat_mat_mul,
        kernel_mat_tmat_mul,
        kernel_mat_sum_rows,
        kernel_mat_sum_cols,
        kernel_softmax_cross_entropy_grad,
        kernel_relu_grad,
        nullptr,//kernel_conv2d_bias_add,
        kernel_update,
        kernel_sigmoid_cross_entropy,
        kernel_sigmoid_cross_entropy_grad,
        nullptr,//kernel_conv2d_input_grad,
        nullptr,//kernel_conv2d_kernel_grad,
        kernel_argmax_acc,
        kernel_moment_update,
        kernel_moment_update2,
        kernel_adam_update,
        kernel_leaky_relu_grad,
        nullptr,//kernel_conv2d_bias_add_grad,
        kernel_tanh_grad,
        nullptr,//kernel_conv2d_transpose,
        nullptr,//kernel_conv2d_transpose_input_grad,
        nullptr,//kernel_conv2d_transpose_kernel_grad
    };

}
