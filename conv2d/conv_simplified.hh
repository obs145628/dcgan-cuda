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







/**
 * Convolution
 * multi channels
 * 1 filter
 * 1 image
 * no stride
 * no padding
 */
Tensor4 conv_cnf1i1s1p0(Tensor4 x, Tensor4 k);


/**
 * Convolution
 * Compute dE/dk, knowing dE/dy (y output of conv)
 * multi-channels
 * 1 filter
 * 1 image
 * no stride
 * no padding
 */
Tensor4 conv_dk_cnf1i1s1p0(Tensor4 x, Tensor4 dy);

/**
 * Convolution
 * Compute dE/dx, knowing dE/dy (y output of conv)
 * multi-channels
 * 1 filter
 * 1 image
 * no stride
 * no padding
 */
Tensor4 conv_dx_cnf1i1s1p0(Tensor4 k, Tensor4 dy);


/**
 * Convolution
 * multi channels
 * multi filters
 * 1 image
 * no stride
 * no padding
 */
Tensor4 conv_cnfni1s1p0(Tensor4 x, Tensor4 k);

/**
 * Convolution
 * Compute dE/dk, knowing dE/dy (y output of conv)
 * multi-channels
 * multi-filters
 * 1 image
 * no stride
 * no padding
 */
Tensor4 conv_dk_cnfni1s1p0(Tensor4 x, Tensor4 dy);

/**
 * Convolution
 * Compute dE/dx, knowing dE/dy (y output of conv)
 * multi-channels
 * multi-filters
 * 1 image
 * no stride
 * no padding
 */
Tensor4 conv_dx_cnfni1s1p0(Tensor4 k, Tensor4 dy);









/**
 * Convolution
 * 1 channels
 * 1 filter
 * 1 image
 * strides
 * no padding
 */
Tensor4 conv_c1f1i1snp0(Tensor4 x, Tensor4 k, std::size_t sh, std::size_t sw);

/**
 * Convolution
 * Compute dE/dk, knowing dE/dy (y output of conv)
 * 1 channels
 * 1 filter
 * 1 image
 * strides
 * no padding
 */
Tensor4 conv_dk_c1f1i1snp0(Tensor4 x, Tensor4 dy, std::size_t sh, std::size_t sw);

/**
 * Convolution
 * Compute dE/dx, knowing dE/dy (y output of conv)
 * 1 channels
 * 1 filter
 * 1 image
 * strides
 * no padding
 */
Tensor4 conv_dx_c1f1i1snp0(Tensor4 k, Tensor4 dy, std::size_t sh, std::size_t sw);









/**
 * Convolution
 * 1 channels
 * 1 filter
 * 1 image
 * strides
 * padding
 */
Tensor4 conv_c1f1i1snpn(Tensor4 x, Tensor4 k, std::size_t sh, std::size_t sw,
                        std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);

/**
 * Convolution
 * Compute dE/dk, knowing dE/dy (y output of conv)
 * 1 channels
 * 1 filter
 * 1 image
 * strides
 * padding
 */
Tensor4 conv_dk_c1f1i1snpn(Tensor4 x, Tensor4 dy, std::size_t sh, std::size_t sw,
                           std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);

/**
 * Convolution
 * Compute dE/dx, knowing dE/dy (y output of conv)
 * 1 channels
 * 1 filter
 * 1 image
 * strides
 * padding
 */
Tensor4 conv_dx_c1f1i1snpn(Tensor4 k, Tensor4 dy, std::size_t sh, std::size_t sw,
                           std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);










/**
 * Convolution
 * multi channels
 * multi filters
 * 1 image
 * strides
 * padding
 */
Tensor4 conv_cnfni1snpn(Tensor4 x, Tensor4 k, std::size_t sh, std::size_t sw,
                        std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);

/**
 * Convolution
 * Compute dE/dk, knowing dE/dy (y output of conv)
 * multi channels
 * multi filters
 * 1 image
 * strides
 * padding
 */
Tensor4 conv_dk_cnfni1snpn(Tensor4 x, Tensor4 dy, std::size_t sh, std::size_t sw,
                           std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);

/**
 * Convolution
 * Compute dE/dx, knowing dE/dy (y output of conv)
 * multi channels
 * multi filters
 * 1 image
 * strides
 * padding
 */
Tensor4 conv_dx_cnfni1snpn(Tensor4 k, Tensor4 dy, std::size_t sh, std::size_t sw,
                           std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);
