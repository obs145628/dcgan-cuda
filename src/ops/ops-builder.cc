#include "ops-builder.hh"
#include <stdexcept>


#include "graph.hh"
#include "argmax-accuracy.hh"
#include "input.hh"
#include "log-softmax.hh"
#include "mat-mat-mul.hh"
#include "mat-mul-add.hh"
#include "mat-rvect-add.hh"
#include "mat-sum.hh"
#include "mse.hh"
#include "mse-grad.hh"
#include "relu-grad.hh"
#include "seq.hh"
#include "sigmoid-cross-entropy.hh"
#include "sigmoid-cross-entropy-grad.hh"
#include "sigmoid-grad.hh"
#include "softmax.hh"
#include "softmax-cross-entropy.hh"
#include "softmax-cross-entropy-grad.hh"
#include "update.hh"
#include "variable.hh"
#include "vect-sigmoid.hh"
#include "conv2d.hh"
#include "conv2d-bias-add.hh"
#include "conv2d-input-grad.hh"
#include "conv2d-kernel-grad.hh"
#include "vect-relu.hh"
#include "vect-relu-leaky.hh"
#include "vect-tanh.hh"
#include "reshape.hh"


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

    ArgmaxAccuracy* OpsBuilder::argmax_accuracy(Op* y, Op* y_hat)
    {
        if (y->shape_get().ndims() != 2)
            throw std::runtime_error {"y must be a matrix"};
        if (y_hat->shape_get().ndims() != 2)
            throw std::runtime_error {"y_hat must be a matrix"};
        if (y->shape_get() != y_hat->shape_get())
            throw std::runtime_error {"y and y_hat must have the same shape"};

        auto res = new ArgmaxAccuracy(y, y_hat);
        graph_.add(res);
        return res;
    }

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

    Conv2DBiasAdd* OpsBuilder::conv2d_bias_add(Op* z, Op* bias)
    {
        if (z->shape_get().ndims() != 4)
            throw std::runtime_error {"Conv2DBiasAdd:z must be a 4D tensor"};
        if (bias->shape_get().ndims() != 1)
            throw std::runtime_error {"Conv2DBiasAdd:bias must be a 1D array"};
        if (z->shape_get()[3] != bias->shape_get()[0])
            throw std::runtime_error {"Conv2DBiasAdd:z and bias shape are not corresponding"};
        auto res = new Conv2DBiasAdd(z, bias);
        graph_.add(res);
        return res;
    }

    Conv2DInputGrad* OpsBuilder::conv2d_input_grad(Op* y, Op* kernel, const int* strides, const int* input_size)
    {
        auto res = new Conv2DInputGrad(y, kernel, strides, input_size);
        graph_.add(res);
        return res;
    }

    Conv2DKernelGrad* OpsBuilder::conv2d_kernel_grad(Op* y, Op* input, const int* strides, const int* kernel_size)
    {
        auto res = new Conv2DKernelGrad(y, input, strides, kernel_size);
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

    MatMatMul* OpsBuilder::mat_mat_mul(Op* left, Op* right, bool left_tr, bool right_tr)
    {
        if (left->shape_get().ndims() != 2)
            throw std::runtime_error{"left operand must be a matrix"};
        if (right->shape_get().ndims() != 2)
            throw std::runtime_error{"right operand must be a matrix"};
        if (left->shape_get()[!left_tr] != right->shape_get()[right_tr])
            throw std::runtime_error{"left[1] and right[0] differ"};

        auto res = new MatMatMul(left, right, left_tr, right_tr);
        graph_.add(res);
        return res;
    }

    MatMulAdd* OpsBuilder::mat_mul_add(Op* x, Op* w, Op* b)
    {
        if (x->shape_get().ndims() != 2)
            throw std::runtime_error{"x must be a matrix"};
        if (w->shape_get().ndims() != 2)
            throw std::runtime_error{"w must be a matrix"};
        if (b->shape_get().ndims() != 1)
            throw std::runtime_error{"b must be a vector"};
        if (x->shape_get()[1] != w->shape_get()[0])
            throw std::runtime_error{"x[1] and w[0] differ"};
        if (w->shape_get()[1] != b->shape_get()[0])
            throw std::runtime_error{"w[1] and b[0] differ"};

        auto res = new MatMulAdd(x, w, b);
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

    MatSum* OpsBuilder::mat_sum(Op* arg, std::size_t axis)
    {
        if (arg->shape_get().ndims() != 2)
            throw std::runtime_error {"arg must be a matrix"};
        if (axis >= 2)
            throw std::runtime_error {"axis must be 0 or 1"};

        auto res = new MatSum(arg, axis);
        graph_.add(res);
        return res;
    }

    MSE* OpsBuilder::mse(Op* y, Op* y_hat)
    {
        if (y->shape_get().ndims() != 2)
            throw std::runtime_error {"MSE:y must be a matrix"};
        if (y_hat->shape_get().ndims() != 2)
            throw std::runtime_error {"MSE:y_hat must be a matrix"};
        if (y->shape_get() != y_hat->shape_get())
            throw std::runtime_error {"MSE: y and y_hat must have the same shape"};

        auto res = new MSE(y, y_hat);
        graph_.add(res);
        return res;
    }

    Reshape* OpsBuilder::reshape(Op* arg, const Shape& shape)
    {
      auto& arg_shape = arg->shape_get();
      if (shape.defined() && shape.total() != arg_shape.total())
          throw std::runtime_error {"Reshape:"};
      //    if (! shape.defined() && (arg_shape.total() % (- shape.total()) != 0))
      //    throw std::runtime_error {"Reshape:"};
      // nb -1 = max 1 ?? has to be checked
      auto res = new Reshape(arg, shape);
      graph_.add(res);
      return res;
    }

    MSEGrad* OpsBuilder::mse_grad(Op* y, Op* y_hat)
    {
        if (y->shape_get().ndims() != 2)
            throw std::runtime_error {"MSEGrad: y must be a matrix"};
        if (y_hat->shape_get().ndims() != 2)
            throw std::runtime_error {"MSEGrad: y_hat must be a matrix"};
        if (y->shape_get() != y_hat->shape_get())
            throw std::runtime_error {"MSEGrad: y and y_hat must have the same shape"};

        auto res = new MSEGrad(y, y_hat);
        graph_.add(res);
        return res;
    }

    ReluGrad* OpsBuilder::relu_grad(Op* z, Op* dout)
    {
        if (z->shape_get() != dout->shape_get())
            throw std::runtime_error {"ReluGrad: z and dout must have the same shape"};

        auto res = new ReluGrad(z, dout);
        graph_.add(res);
        return res;
    }

    Seq* OpsBuilder::seq(const std::vector<Op*>& ops)
    {
        if (ops.empty())
            throw std::runtime_error {"seq: ops can't be empty"};
        auto res = new Seq(ops);
        graph_.add(res);
        return res;
    }

    SigmoidCrossEntropy* OpsBuilder::sigmoid_cross_entropy(Op* y, Op* logits)
    {
        if (y->shape_get().ndims() != 2)
            throw std::runtime_error {"y must be a matrix"};
        if (logits->shape_get().ndims() != 2)
            throw std::runtime_error {"logits must be a matrix"};
        if (y->shape_get() != logits->shape_get())
            throw std::runtime_error {"y and logits must have the same shape"};
            
        auto res = new SigmoidCrossEntropy(y, logits);
        graph_.add(res);
        return res;
    }

    SigmoidCrossEntropyGrad* OpsBuilder::sigmoid_cross_entropy_grad(Op* y, Op* logits)
    {
        if (y->shape_get().ndims() != 2)
            throw std::runtime_error {"y must be a matrix"};
        if (logits->shape_get().ndims() != 2)
            throw std::runtime_error {"logits must be a matrix"};
        if (y->shape_get() != logits->shape_get())
            throw std::runtime_error {"y and logits must have the same shape"};
        
        auto res = new SigmoidCrossEntropyGrad(y, logits);
        graph_.add(res);
        return res;
    }

    SigmoidGrad* OpsBuilder::sigmoid_grad(Op* sig_out, Op* dout)
    {
        if (sig_out->shape_get() != dout->shape_get())
            throw std::runtime_error {"SigmoidGrad: sig_out and dout must have the same shape"};

        auto res = new SigmoidGrad(sig_out, dout);
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
            throw std::runtime_error {"y must be a matrix"};
        if (logits->shape_get().ndims() != 2)
            throw std::runtime_error {"logits must be a matrix"};
        if (y->shape_get() != logits->shape_get())
            throw std::runtime_error {"y and logits must have the same shape"};
            
        auto res = new SoftmaxCrossEntropy(y, logits);
        graph_.add(res);
        return res;
    }

    SoftmaxCrossEntropyGrad* OpsBuilder::softmax_cross_entropy_grad(Op* y, Op* logits)
    {
        if (y->shape_get().ndims() != 2)
            throw std::runtime_error {"y must be a matrix"};
        if (logits->shape_get().ndims() != 2)
            throw std::runtime_error {"logits must be a matrix"};
        if (y->shape_get() != logits->shape_get())
            throw std::runtime_error {"y and logits must have the same shape"};
        
        auto res = new SoftmaxCrossEntropyGrad(y, logits);
        graph_.add(res);
        return res;
    }

    Update* OpsBuilder::update(Variable* var, Op* dt, Op* coeff)
    {
        if (var->shape_get() != dt->shape_get())
            throw std::runtime_error {"var and dt must have the same shape"};
        if (coeff->shape_get().ndims())
            throw std::runtime_error {"coeff must be a scalar"};

        auto res = new Update(var, dt, coeff);
        graph_.add(res);
        return res;
    }

    Variable* OpsBuilder::variable(const Shape& shape, bool trainable)
    {
        if (!shape.defined())
            throw std::runtime_error{"shape not fully defined"};
        auto res = new Variable(shape, trainable);
        graph_.add_var(res);
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

    VectReluLeaky* OpsBuilder::vect_relu_leaky(Op* arg, const dbl_t alpha)
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
