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
        kernel_tanh
    };
}
