#include "cost-function.hh"

dbl_t QuadraticCost::cost()
{
    std::size_t len = rows_ * cols_;
    dbl_t res = 0;
    for (std::size_t i = 0; i < len; ++i)
	res += (in_[i] - in_hat_[i]) * (in_[i] - in_hat_[i]);
    return res / len;
}

void QuadraticCost::cost_grad()
{
    std::size_t len = rows_ * cols_;
    dbl_t coeff = 2.0 / len;
    for (std::size_t i = 0; i < len; ++i)
	out_grad_[i] = coeff * (in_hat_[i] - in_[i]);
}
