#pragma once

#include "../memory/types.hh"

class Activation
{
public:

    virtual ~Activation() = default;

    /**
     * Compute the activation value
     * Will be replaced by it's own kernel later
     */
    virtual void compute(const dbl_t* in_begin, const dbl_t* in_end,
			 dbl_t* out_begin) = 0;

    /**
     * Compute the gradient of the activation value
     * Will be replaced by it's own kernel
     */
    virtual void compute_prime(const dbl_t* in_begin, const dbl_t* in_end,
			       dbl_t* out_begin) = 0;
};

/**
 * Implement the sigmoid activation function
 */
class SigmoidActivation : public Activation
{
public:
    virtual ~SigmoidActivation() = default;
    void compute(const dbl_t* in_begin, const dbl_t* in_end,
		 dbl_t* out_begin) override;
    void compute_prime(const dbl_t* in_begin, const dbl_t* in_end,
		       dbl_t* out_begin) override;
};
