#include "cost-function.hh"


const char* QuadraticCost::name() const
{
    return "quadratic";
}

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

/*
inline Vector QuadraticCost::cost_prime(const Vector& y, const Vector& y_hat) const
{
    return y_hat - y;
}

inline const char* CrossEntropyCost::name() const
{
    return "cross-entropy";
}

inline num_t CrossEntropyCost::cost(const Vector& y, const Vector& y_hat) const
{
    num_t res = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
    {
	//TODO: handle numerical stability
	num_t val = -y[i] * std::log(y_hat[i]) - (1 - y[i]) * std::log(1 - y_hat[i]);
	res += val;
    }
    return res;
}

inline Vector CrossEntropyCost::cost_prime(const Vector& y, const Vector& y_hat) const
{
    Vector div = y_hat * (1 - y_hat);
    return (y_hat - y) / div;
}
*/
