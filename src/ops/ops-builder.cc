#include "ops-builder.hh"
#include <stdexcept>

#include "graph.hh"
#include "input.hh"
#include "log-softmax.hh"
#include "mat-mat-mul.hh"
#include "mat-rvect-add.hh"
#include "mse.hh"
#include "softmax.hh"
#include "softmax-cross-entropy.hh"
#include "variable.hh"
#include "vect-sigmoid.hh"
#include "conv2d.hh"
#include "vect-relu.hh"
#include "vect-relu-leaky.hh"
#include "vect-tanh.hh"


namespace ops
{

    OpsBuilder& OpsBuilder::instance()
    {
        static OpsBuilder builder;
        return builder;
    }

    OpsBuilder::OpsBuilder()
        : graph_(Graph::instance())
    {}

    Conv2D* OpsBuilder::conv2d(Op* input, Op* kernel, const int* strides)
    {
        if (input->shape_get().ndims() != 4)
            throw std::runtime_error {"Conv2D:input must be a 4D tensor"};
        if (kernel->shape_get().ndims() != 4)
            throw std::runtime_error {"Conv2D:kernel must be a 4D tensor"};
        auto res = new Conv2D(input, kernel, strides);
        graph_.add(res);
        return res;
    }
    
    Input* OpsBuilder::input(const Shape& shape)
    {
        auto res = new Input(shape);
        graph_.add(res);
        return res;
    }

    LogSoftmax* OpsBuilder::log_softmax(Op* arg)
    {
        if (arg->shape_get().ndims() != 2)
            throw std::runtime_error{"log softmax input must be a matrix"};

        auto res = new LogSoftmax(arg);
        graph_.add(res);
        return res;
    }

    MatMatMul* OpsBuilder::mat_mat_mul(Op* left, Op* right)
    {
        if (left->shape_get().ndims() != 2)
            throw std::runtime_error{"left operand must be a matrix"};
        if (right->shape_get().ndims() != 2)
            throw std::runtime_error{"right operand must be a matrix"};
        if (left->shape_get()[1] != right->shape_get()[0])
            throw std::runtime_error{"left[1] and right[0] differ"};

        auto res = new MatMatMul(left, right);
        graph_.add(res);
        return res;
    }

    MatRvectAdd* OpsBuilder::mat_rvect_add(Op* left, Op* right)
    {
        if (left->shape_get().ndims() != 2)
            throw std::runtime_error{"left operand must be a matrix"};
        if (right->shape_get().ndims() != 1)
            throw std::runtime_error{"right operand must be a vector"};
        if (left->shape_get()[1] != right->shape_get()[0])
            throw std::runtime_error{"left[1] and right[0] differ"};

        auto res = new MatRvectAdd(left, right);
        graph_.add(res);
        return res;
    }
    
    MSE* OpsBuilder::mse(Op* y, Op* y_hat)
    {
        if (y->shape_get().ndims() != 2)
            throw std::runtime_error {"MSE:y must be a matrix"};
        if (y_hat->shape_get().ndims() != 2)
            throw std::runtime_error {"MSE:y_hat must be a matrix"};
        auto res = new MSE(y, y_hat);
        graph_.add(res);
        return res;
    }

    Softmax* OpsBuilder::softmax(Op* arg)
    {
        if (arg->shape_get().ndims() != 2)
            throw std::runtime_error{"softmax input must be a matrix"};

        auto res = new Softmax(arg);
        graph_.add(res);
        return res;
    }

    SoftmaxCrossEntropy* OpsBuilder::softmax_cross_entropy(Op* y, Op* logits)
    {
        if (y->shape_get().ndims() != 2)
            throw std::runtime_error {"CrossEntropy:y must be a matrix"};
        if (logits->shape_get().ndims() != 2)
            throw std::runtime_error {"CrossEntropy:logits must be a matrix"};
        auto res = new SoftmaxCrossEntropy(y, logits);
        graph_.add(res);
        return res;
    }

    Variable* OpsBuilder::variable(const Shape& shape)
    {
        if (!shape.defined())
            throw std::runtime_error{"shape not fully defined"};
        auto res = new Variable(shape);
        graph_.add(res);
        return res;
    }

    VectSigmoid* OpsBuilder::vect_sigmoid(Op* arg)
    {
        auto res = new VectSigmoid(arg);
        graph_.add(res);
        return res;
    }

    VectRelu* OpsBuilder::vect_relu(Op* arg)
    {
        auto res = new VectRelu(arg);
        graph_.add(res);
        return res;
    }

    VectReluLeaky* OpsBuilder::vect_relu_leaky(Op* arg, const dbl_t alpha = 0.2)
    {
        auto res = new VectReluLeaky(arg, alpha);
        graph_.add(res);
        return res;
    }

    VectTanh* OpsBuilder::vect_tanh(Op* arg)
    {
        auto res = new VectTanh(arg);
        graph_.add(res);
        return res;
    }
}
