#include "tensor4.hh"

/**
 * Compute a single value of the convolution
 * input (i1 * h1 * w1 * c1) nb input * input height * input width * input channels
 * filter (fh * fw * c1 * k) filter height * filter with * input channels * nb filters
 * d1 - image to work with (0 <= d1 < i1)
 * d2 - row to start convolution (0 <= d2 < i2 - fh)
 * d3 - col to start convolution (0 <= d3 < i3 - fw)
 * d4 - index of ouput channel
 */
float compute_val(const Tensor4& input, const Tensor4& filter,
                  std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4);


/**
 * Compute a full convolution without padding
 * input (i1, h1, w1, c1) nb input * input height * input width * input channels
 * filter (fh, fw, c1, k) filter height * filter with * input channels * nb filters
 * sh - height of strides
 * sw - width of strides
 * out (i1, (h1 - fh) / sh + 1, (w1 - fw) / sw + 1, k)
 */
Tensor4 conv_no_pad(const Tensor4& input, const Tensor4& filter,
                    std::size_t sh, std::size_t sw);

/**
 * Compute a full convolution with padding
 * input (i1, h1, w1, c1) nb input * input height * input width * input channels
 * filter (fh, fw, c1, k) filter height * filter with * input channels * nb filters
 * sh - height of strides
 * sw - width of strides
 * ph - vertical padding
 * pw - horizontal padding
 * out (i1, (h1 - fh + 2ph) / sh + 1, (w1 - fw + 2pw) / sw + 1, k)
 */
Tensor4 conv_pad(const Tensor4& input, const Tensor4& filter,
                 std::size_t sh, std::size_t sw,
                 std::size_t ph, std::size_t pw);
