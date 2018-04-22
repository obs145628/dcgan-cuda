#pragma once

#include "fwd.hh"
#include "../memory/types.hh"

/**
 * Network layer class
 */
class Layer
{

public:

    virtual ~Layer() = default;

    /**
     * Input tensor
     */
    const dbl_t* input() const;

    /**
     * Output tensor
     */
    dbl_t* output() const;

    void input_set(const dbl_t* input);

    void output_set(dbl_t* output);

    virtual void clean_tensors() = 0;
    virtual void ninputs_set(std::size_t ninputs) = 0;
    virtual std::size_t input_size() const = 0;
    virtual std::size_t output_size() const = 0;
    

    /**
     * Apply forward propgation on the tensor in input
     * Store results tensor in output
     */
    virtual void forward() = 0;

protected:
    const dbl_t* input_ = nullptr;
    dbl_t* output_ = nullptr;
};

#include "layer.hxx"
