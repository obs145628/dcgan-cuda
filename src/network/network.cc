#include "network.hh"

#include "layer.hh"
#include "cost-function.hh"
#include "../memory/alloc.hh"
#include "../memory/copy.hh"


Network::Network(const layers_t& layers, CostFunction* cost)
    : layers_(layers)
    , cost_(cost)
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
    tensor_write(input_, input_ + len, input);
    forward();
}

dbl_t Network::compute_loss(std::size_t ninputs, const dbl_t* x, const dbl_t* y)
{
    forward(ninputs, x);
    tensor_write(cost_->in_get(), cost_->in_get() + + ninputs * layers_.back()->output_size(), y);
    return cost_->cost();
}
