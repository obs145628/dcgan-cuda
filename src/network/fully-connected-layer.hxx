#pragma once

#include "fully-connected-layer.hh"

inline dbl_t* FullyConnectedLayer::w_get() const
{
    return w_;
}

inline dbl_t* FullyConnectedLayer::b_get() const
{
    return b_;
}

inline dbl_t* FullyConnectedLayer::z_get() const
{
    return z_;
}
