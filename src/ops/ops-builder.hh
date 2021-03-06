#pragma once

#include <cstddef>
#include <vector>
#include "fwd.hh"
#include "../memory/types.hh"

namespace ops
{

    class OpsBuilder
    {

    public:
        static OpsBuilder& instance();

        OpsBuilder(const OpsBuilder&) = delete;
        OpsBuilder(OpsBuilder&&) = delete;
        OpsBuilder& operator=(const OpsBuilder&) = delete;
        OpsBuilder& operator=(OpsBuilder&&) = delete;

        Add* add(Op* left, Op* right);
        AdamUpdate* adam_update(Variable* var, Op* m, Op* v,
                                dbl_t learning_rate,
                                dbl_t beta1, dbl_t beta2, dbl_t eps);
        ArgmaxAccuracy* argmax_accuracy(Op* y, Op* y_hat);
        Conv2D* conv2d(Op* input, Op* kernel, const int* strides);
        Conv2DBiasAdd* conv2d_bias_add(Op* z, Op* bias);
        Conv2DBiasAddGrad* conv2d_bias_add_grad(Op* z);
        Conv2DInputGrad* conv2d_input_grad(Op* y, Op* kernel, const int* strides, const int* input_size);
        Conv2DKernelGrad* conv2d_kernel_grad(Op* y, Op* input, const int* strides, const int* kernel_size, const int* padded_size);
        Conv2DTranspose* conv2d_transpose(Op* input, Op* kernel, const int* out_size, const int* strides);
        Conv2DTransposeInputGrad* conv2d_transpose_input_grad(Op* y, Op* kernel, const int* strides, const int* input_size);
        Conv2DTransposeKernelGrad* conv2d_transpose_kernel_grad(Op* y, Op* input, const int* strides, const int* kernel_size);
        Input* input(const Shape& shape);
        LeakyReluGrad* leaky_relu_grad(Op* z, Op* dout, dbl_t alpha);
        LogSoftmax* log_softmax(Op* arg);
        MatMatMul* mat_mat_mul(Op* left, Op* right, bool left_tr = false, bool right_tr = false);
        MatMulAdd* mat_mul_add(Op* x, Op* w, Op* b);
        MatRvectAdd* mat_rvect_add(Op* left, Op* right);
        MatSum* mat_sum(Op* arg, std::size_t axis);
        MomentUpdate* moment_update(Variable* var, Op* dt,
                                    dbl_t coeff1, dbl_t coeff2, bool sq_update);
        MSE* mse(Op* y, Op* y_hat);
        MSEGrad* mse_grad(Op* y, Op* y_hat);
        ReluGrad* relu_grad(Op* z, Op* dout);
        Seq* seq(const std::vector<Op*>& ops);
        SigmoidCrossEntropy* sigmoid_cross_entropy(Op* y, Op* logits);
        SigmoidCrossEntropyGrad* sigmoid_cross_entropy_grad(Op* y, Op* logits);
        SigmoidGrad* sigmoid_grad(Op* sig_out, Op* dout);
        Softmax* softmax(Op* arg);
        SoftmaxCrossEntropy* softmax_cross_entropy(Op* y, Op* logits);
        SoftmaxCrossEntropyGrad* softmax_cross_entropy_grad(Op* y, Op* logits);
        TanhGrad* tanh_grad(Op* tanh_out, Op* dout);
        Update* update(Variable* var, Op* dt, Op* coeff);
        Variable* variable(const Shape& shape, bool trainable = false);
        VectSigmoid* vect_sigmoid(Op* arg);
        VectRelu* vect_relu(Op* arg);
        VectReluLeaky* vect_relu_leaky(Op* arg, const dbl_t alpha = 0.2);
        VectTanh* vect_tanh(Op* arg);
        Reshape* reshape(Op* arg, const Shape& shape);

    private:
        OpsBuilder();

        Graph& graph_;

    };

}
