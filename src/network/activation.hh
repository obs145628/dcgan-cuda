#pragma once

#include "../memory/types.hh"

class Activation
{
public:
    virtual ~Activation() = default;
    virtual const char* name() const = 0;

    virtual void compute(const dbl_t* in_begin, const dbl_t* in_end,
			 dbl_t* out_begin) = 0;

    virtual void compute_prime(const dbl_t* in_begin, const dbl_t* in_end,
			       dbl_t* out_begin) = 0;
};

class SigmoidActivation : public Activation
{
public:
    virtual ~SigmoidActivation() = default;
    const char* name() const override;
    void compute(const dbl_t* in_begin, const dbl_t* in_end,
		 dbl_t* out_begin) override;
    void compute_prime(const dbl_t* in_begin, const dbl_t* in_end,
		       dbl_t* out_begin) override;
};
