#pragma once

#include "ops.hh"

#include <cmath>


inline dbl_t sigmoid(dbl_t x)
{
    //return x;
    return 1.0 / (1.0 + std::exp(-x));
}


inline dbl_t sigmoid_prime(dbl_t x)
{
    return std::exp(-x) / ((1.0 + std::exp(-x) * (1.0 + std::exp(-x))));
}
