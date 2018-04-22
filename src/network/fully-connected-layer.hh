#pragma once

#include "layer.hh"

/**
 * Dense or fully connected layer
 */
class FullyConnectedLayer : public Layer
{

public:

    /**
     * @param in_size: number of input neurons
     * @param out_size: numbr of output neurons
     * @param activ: activation function
     */
    FullyConnectedLayer(std::size_t in_size, std::size_t out_size,
	Activation* activ);
    virtual ~FullyConnectedLayer();


    virtual void clean_tensors() override;
    virtual void ninputs_set(std::size_t ninputs) override;
    virtual std::size_t input_size() const override;
    virtual std::size_t output_size() const override;
    virtual void forward() override;

    /**
     * Weights matrix
     * (in_size * out_size)
     */
    dbl_t* w_get() const;

    /**
     * Bias vector
     * (out_size)
     */
    dbl_t* b_get() const;

    /**
     * z - intermediate result matrix
     * (ninputs * out_size)
     */
    dbl_t* z_get() const;

private:
    std::size_t ninputs_;
    std::size_t in_size_;
    std::size_t out_size_;
    Activation* activ_;
    
    dbl_t* w_;
    dbl_t* b_;
    dbl_t* z_;
};

#include "fully-connected-layer.hxx"
