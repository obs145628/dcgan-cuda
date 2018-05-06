#pragma once

#include "fwd.hh"

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

	Input* input(const Shape& shape);
	LogSoftmax* log_softmax(Op* arg);
	MatMatMul* mat_mat_mul(Op* left, Op* right);
	MatRvectAdd* mat_rvect_add(Op* left, Op* right);
	Softmax* softmax(Op* arg);
	SoftmaxCrossEntropy* softmax_cross_entropy(Op* y, Op* logits);
	Variable* variable(const Shape& shape);
	VectSigmoid* vect_sigmoid(Op* arg);
	MSE* mse(Op* y, Op* y_hat);
  Conv2D* conv2d(Op* input, Op* kernel, const int* strides);

    private:
	OpsBuilder();

	Graph& graph_;
	
    };
    
}
