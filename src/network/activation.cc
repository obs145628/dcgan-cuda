#include "activation.hh"
#include "../math_cpu/ops.hh"

void SigmoidActivation::compute(const dbl_t* in_begin, const dbl_t* in_end,
				dbl_t* out_begin)
{
    while (in_begin != in_end)
	*out_begin++ = sigmoid(*in_begin++);
}

void SigmoidActivation::compute_prime(const dbl_t* in_begin, const dbl_t* in_end,
				      dbl_t* out_begin)
{
    while (in_begin != in_end)
	*out_begin++ = sigmoid_prime(*in_begin++);
}
