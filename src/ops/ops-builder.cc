#include "ops-builder.hh"
#include <stdexcept>

#include "graph.hh"
#include "input.hh"
#include "mat-mat-mul.hh"
#include "mat-rvect-add.hh"
#include "mse.hh"
#include "softmax.hh"
#include "variable.hh"
#include "vect-sigmoid.hh"

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

    Input* OpsBuilder::input(const Shape& shape)
    {
	auto res = new Input(shape);
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

    Softmax* OpsBuilder::softmax(Op* arg)
    {
	if (arg->shape_get().ndims() != 2)
	    throw std::runtime_error{"softmax input must be a matrix"};

	auto res = new Softmax(arg);
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
}
