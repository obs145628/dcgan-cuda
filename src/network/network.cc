#include "network.hh"

#include <algorithm>

#include "layer.hh"
#include "cost-function.hh"
#include "memory.hh"
//#include "la/tensor_ops.hh"
//#include "dnn/activation.hh"
//#include "dnn/debug-tensors.hh"
//#include "dnn/optimizer.hh"

/*
namespace
{
    constexpr num_t ERR_BACKPROP = 1e-2;
}
*/

Network::Network(const layers_t& layers, CostFunction* cost,
		 test_f test_function)
    : layers_(layers)
    , cost_(cost)
    , test_function_(test_function)
    , ninputs_(0)
    , input_(nullptr)
    , output_(nullptr)
{}

Network::~Network()
{
    clean_tensors();
    for (auto l : layers_)
	delete l;
    delete cost_;
}

layers_t& Network::layers_get()
{
    return layers_;
}

const layers_t& Network::layers_get() const
{
    return layers_;
}

dbl_t* Network::output_get() const
{
    return output_;
}

void Network::clean_tensors()
{
    ninputs_ = 0;

    tensor_free(input_);
    for (auto& l : layers_)
    {
	tensor_free(l->output());
	l->clean_tensors();
    }

    tensor_free(cost_->in_get());
    tensor_free(cost_->out_grad_get());
}

std::size_t Network::ninputs_get() const
{
    return ninputs_;
}

void Network::ninputs_set(std::size_t ninputs)
{
    if (ninputs <= ninputs_)
    {
	ninputs_ = ninputs;
	return;
    }

    clean_tensors();
    ninputs_ = ninputs;

    dbl_t* input = tensor_alloc(ninputs * layers_.front()->input_size());
    input_ = input;

    for (auto& l : layers_)
    {
	dbl_t* next = tensor_alloc(ninputs * l->output_size());
	l->input_set(input);
	l->output_set(next);
	l->ninputs_set(ninputs);
	input = next;
    }
    
    output_ = input;

    dbl_t* cost_in = tensor_alloc(ninputs * layers_.back()->output_size());
    dbl_t* cost_grad = tensor_alloc(ninputs * layers_.back()->output_size());
    cost_->in_set(cost_in);
    cost_->in_hat_set(output_);
    cost_->out_grad_set(cost_grad);
    cost_->rows_set(ninputs);
    cost_->cols_set(layers_.back()->output_size());
}

void Network::forward()
{
    for (auto l : layers_)
        l->forward();
}

void Network::forward(std::size_t ninputs, const dbl_t* input)
{
    ninputs_set(ninputs);
    std::size_t len = ninputs * layers_.front()->input_size();
    std::copy(input, input + len, input_);
    forward();
}

dbl_t Network::compute_loss(std::size_t ninputs, const dbl_t* x, const dbl_t* y)
{
    forward(ninputs, x);
    std::copy(y, y + ninputs * layers_.back()->output_size(), cost_->in_get());
    return cost_->cost();
}



/*
void Network::backpropagation(const Vector& x, const Vector& y)
{
    Vector y_hat = forward(x);
    Vector da = cost_->cost_prime(y, y_hat);
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
	da = (*it)->backpropagation(da);
}

num_t Network::data_cost(const Matrix& x, const Matrix& y)
{
    std::size_t n = x.rows();
    num_t res = 0;

    for (std::size_t i = 0; i < n; ++i)
    {
	Vector y_hat = forward(x[i]);
	res += cost_->cost(y[i], y_hat);
    }

    res /= num_t(n);
    return res;
}


void Network::evaluate(const Matrix& x, const Matrix& y)
{
    std::size_t n = x.rows();
    std::size_t valid = 0;

    for (std::size_t i = 0; i < n; ++i)
	valid += test_function_(forward(x[i]), y[i]);
    
    double acc = (valid * 100.0) / n;
    std::cout << "Evalutation: " << valid << "/" << n << " (" << acc << "%)\n";
}

void Network::train(Matrix& x_train, Matrix& y_train, Matrix& x_test, Matrix& y_test,
		    Optimizer& opti, std::size_t epochs,
		    bool disp_train_cost)
{
    for (std::size_t i = 1; i <= epochs; ++i)
    {
	std::cout << "Epoch: " << i << ": " << std::endl;
	opti.run(*this, x_train, y_train);
	evaluate(x_test, y_test);

	if (disp_train_cost)
	    std::cout << "Train cost: " << data_cost(x_train, y_train) << std::endl;
    }
}


bool Network::check_backpropagation(const Matrix& x, const Matrix& y)
{
    bool valid = true;
    for (std::size_t i = 0; i < x.rows(); ++i)
	backpropagation(x[i], y[i]);

    for (std::size_t i = 0; i < layers_.size(); ++i)
    {
	std::cout << "Layer " << (i + 1) << ": " << std::endl;
	num_t val = layers_[i]->check_grad(*this, x, y);
	bool succ = val < ERR_BACKPROP;
	if (!succ)
	{
	    std::cout << "Invalid layer" << std::endl;
	    valid = false;
	}
    }

    return valid;
}
*/
