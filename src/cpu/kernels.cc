#include "kernels.hh"
#include "ops.hh"
#include "../runtime/node.hh"

namespace cpu
{

    namespace
    {

        void kernel_conv2d(rt::Node* node)
        {
            conv2d(node->in1, node->in2, node->out1, node->intconst, node->sizes1, node->sizes2);
        }

        void kernel_conv2d_bias_add(rt::Node* node)
        {
            conv2d_bias_add(node->in1, node->in2, node->out1, node->sizes1);
        }

        void kernel_conv2d_input_grad(rt::Node* node)
        {
            conv2d_input_grad(node->in1, node->in2, node->intconst[0], node->sizes1, node->sizes2, node->out1);
        }

        void kernel_conv2d_kernel_grad(rt::Node* node)
        {
            conv2d_kernel_grad(node->in1, node->in2, node->intconst[0], node->sizes1, node->sizes2, node->out1);
        }

        void kernel_mat_mat_mul(rt::Node* node)
        {
            mm_mul(node->in1, node->in2, node->out1,
                   node->len1, node->len2, node->len3);
        }

        void kernel_mat_rvect_add(rt::Node* node)
        {
            mvrow_add(node->in1, node->in2, node->out1,
                      node->len1, node->len2);
        }

        void kernel_relu(rt::Node* node)
        {
            vect_relu(node->in1, node->out1, node->len1);
        }

        void kernel_relu_leaky(rt::Node* node)
        {
            vect_relu_leaky(node->in1, node->out1, node->len1, node->alpha_leaky);
        }

        void kernel_sigmoid(rt::Node* node)
        {
            vect_sigmoid(node->in1, node->out1, node->len1);
        }

        void kernel_mse(rt::Node* node)
        {
            *node->out1 = mse(node->in1, node->in2, node->len1, node->len2);
        }

        void kernel_softmax(rt::Node* node)
        {
            softmax(node->in1, node->out1, node->len1, node->len2);
        }

        void kernel_log_softmax(rt::Node* node)
        {
            log_softmax(node->in1, node->out1, node->len1, node->len2);
        }

        void kernel_softmax_cross_entropy(rt::Node* node)
        {
            *node->out1 = softmax_cross_entropy(node->in1, node->in2, node->len1, node->len2);
        }

        void kernel_tanh(rt::Node* node)
        {
            vect_tanh(node->in1, node->out1, node->len1);
        }

        void kernel_mse_grad(rt::Node* node)
        {
            vect_sub_coeff(node->in2, node->in1, 2. / node->len1, node->out1, node->len1);
        }

        void kernel_sigmoid_grad(rt::Node* node)
        {
            sigmoid_grad(node->in1, node->in2, node->out1, node->len1);
        }

        void kernel_mat_mul_add(rt::Node* node)
        {
            mat_mul_add(node->in1, node->in2, node->in3, node->out1,
                        node->len1, node->len2, node->len3);
        }

        void kernel_tmat_mat_mul(rt::Node* node)
        {
            tmm_mul(node->in1, node->in2, node->out1,
                    node->len1, node->len2, node->len3);
        }

        void kernel_mat_tmat_mul(rt::Node* node)
        {
            mtm_mul(node->in1, node->in2, node->out1,
                    node->len1, node->len2, node->len3);
        }

        void kernel_mat_sum_rows(rt::Node* node)
        {
            mat_sum_rows(node->in1, node->out1, node->len1, node->len2);
        }

        void kernel_mat_sum_cols(rt::Node* node)
        {
            mat_sum_cols(node->in1, node->out1, node->len1, node->len2);
        }

        void kernel_softmax_cross_entropy_grad(rt::Node* node)
        {
            softmax_cross_entropy_grad(node->in1, node->in2, node->out1, node->len1, node->len2);
        }

        void kernel_relu_grad(rt::Node* node)
        {
            relu_grad(node->in1, node->in2, node->out1, node->len1);
        }

        void kernel_update(rt::Node* node)
        {
            vect_update(node->in1, node->out1, *node->in2, node->len1);
        }

        void kernel_sigmoid_cross_entropy(rt::Node* node)
        {
            *(node->out1) = sigmoid_cross_entropy(node->in1, node->in2, node->len1);
        }

        void kernel_sigmoid_cross_entropy_grad(rt::Node* node)
        {
            sigmoid_cross_entropy_grad(node->in1, node->in2, node->out1, node->len1);
        }

        void kernel_argmax_acc(rt::Node* node)
        {
            *(node->out1) = argmax_acc(node->in1, node->in2, node->len1, node->len2); 
        }

        void kernel_moment_update(rt::Node* node)
        {
            moment_update(node->in1, node->out1, node->cons1, node->cons2, node->len1);
        }

        void kernel_moment_update2(rt::Node* node)
        {
            moment_update2(node->in1, node->out1, node->cons1, node->cons2, node->len1);
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

            adam_update(node->in1, node->in2, node->out1, lrt, eps, node->len1);
        }

        void kernel_leaky_relu_grad(rt::Node* node)
        {
            leaky_relu_grad(node->in1, node->in2, node->out1, node->cons1, node->len1);
        }

    }

    kernel_f kernels_list[64] = {
        kernel_mat_mat_mul,
        kernel_mat_rvect_add,
        kernel_sigmoid,
        kernel_mse,
        kernel_softmax,
        kernel_log_softmax,
        kernel_softmax_cross_entropy,
        kernel_conv2d,
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
        kernel_conv2d_bias_add,
        kernel_update,
        kernel_sigmoid_cross_entropy,
        kernel_sigmoid_cross_entropy_grad,
        kernel_conv2d_input_grad,
        kernel_conv2d_kernel_grad,
        kernel_argmax_acc,
        kernel_moment_update,
        kernel_moment_update2,
        kernel_adam_update,
        kernel_leaky_relu_grad,
    };
}
