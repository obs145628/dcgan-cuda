#pragma once

#include "tensor4.hh"

/**
 * Convolution
 * 1 channels
 * 1 filter
 * 1 image
 * no stride
 * no padding
 */
Tensor4 conv_c1f1i1s1p0(Tensor4 x, Tensor4 k);

/**
 * Convolution
 * Compute dE/dk, knowing dE/dy (y output of conv)
 * 1 channels
 * 1 filter
 * 1 image
 * no stride
 * no padding
 */
Tensor4 conv_dk_c1f1i1s1p0(Tensor4 x, Tensor4 dy);

/**
 * Convolution
 * Compute dE/dx, knowing dE/dy (y output of conv)
 * 1 channels
 * 1 filter
 * 1 image
 * no stride
 * no padding
 */
Tensor4 conv_dx_c1f1i1s1p0(Tensor4 k, Tensor4 dy);
