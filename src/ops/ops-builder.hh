#pragma once

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

        Conv2D* conv2d(Op* input, Op* kernel, const int* strides);
        Input* input(const Shape& shape);
        LogSoftmax* log_softmax(Op* arg);
        MatMatMul* mat_mat_mul(Op* left, Op* right);
        MatRvectAdd* mat_rvect_add(Op* left, Op* right);
        MSE* mse(Op* y, Op* y_hat);
        MSEGrad* mse_grad(Op* y, Op* y_hat);
        SigmoidGrad* sigmoid_grad(Op* sig_out, Op* dout);
        Softmax* softmax(Op* arg);
        SoftmaxCrossEntropy* softmax_cross_entropy(Op* y, Op* logits);
        Variable* variable(const Shape& shape);
        VectSigmoid* vect_sigmoid(Op* arg);
        VectRelu* vect_relu(Op* arg);
        VectReluLeaky* vect_relu_leaky(Op* arg, const dbl_t alpha = 0.2);
        VectTanh* vect_tanh(Op* arg);

    private:
        OpsBuilder();

        Graph& graph_;

    };

}
