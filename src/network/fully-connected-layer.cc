#include "fully-connected-layer.hh"
#include "memory.hh"
#include "activation.hh"
#include "../math_cpu/mat.hh"

FullyConnectedLayer::FullyConnectedLayer(std::size_t in_size, std::size_t out_size,
					 Activation* activ)
    : ninputs_(0)
    , in_size_(in_size)
    , out_size_(out_size)
    , activ_(activ)
    , w_(tensor_alloc(in_size * out_size))
    , b_(tensor_alloc(out_size))
    , z_(nullptr)
{
    for (std::size_t i = 0; i < in_size_ * out_size_; ++i)
	w_[i] = 0;

    for (std::size_t i = 0; i < out_size_; ++i)
	b_[i] = 0;
}

FullyConnectedLayer::~FullyConnectedLayer()
{
    delete activ_;
    tensor_free(w_);
    tensor_free(b_);
}


void FullyConnectedLayer::clean_tensors()
{
    ninputs_ = 0;
    tensor_free(z_);
}

void FullyConnectedLayer::ninputs_set(std::size_t ninputs)
{
    ninputs_ = ninputs;
    z_ = tensor_alloc(out_size_ * ninputs);
}

std::size_t FullyConnectedLayer::input_size() const
{
    return in_size_;
}

std::size_t FullyConnectedLayer::output_size() const
{
    return out_size_;
}

void FullyConnectedLayer::forward()
{
    mm_mul(input_, w_, z_, ninputs_, in_size_, out_size_);
    mvrow_add(z_, b_, z_, ninputs_, out_size_);
    activ_->compute(z_, z_ + ninputs_ * out_size_, output_);
}
