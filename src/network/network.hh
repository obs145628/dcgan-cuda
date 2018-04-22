#pragma once

#include "fwd.hh"
#include "../memory/types.hh"

#include <iostream>

class Network
{

public:
    
    Network(const layers_t& layers, CostFunction* cost);
    ~Network();

    
    layers_t& layers_get();
    const layers_t& layers_get() const;

    /**
     * Return the output tensor
     */
    dbl_t* output_get() const;

    /**
     * Free all allocated memory for the network
     */
    void clean_tensors();

    /**
     * Number of examples passed to the network
     */
    std::size_t ninputs_get() const;
    void ninputs_set(std::size_t ninputs);

    /**
     * Apply a forward pass through all layers
     * Takes ninputs examples at the same time
     * Input data is in input
     * Result is stored in output
     */
    void forward();

    /**
     * - resize the network to take ninputs
     * - copy data to input
     * - run a forward pass
     */
    void forward(std::size_t ninputs, const dbl_t* input);

    /**
     * - run a forward pass of x
     * - compute the loss between x and y
     */
    dbl_t compute_loss(std::size_t ninputs, const dbl_t* x, const dbl_t* y);
    
private:
    layers_t layers_;
    CostFunction* cost_;
    std::size_t ninputs_;

    dbl_t* input_;
    dbl_t* output_;
};

