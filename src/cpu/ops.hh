#pragma once

#include "../memory/types.hh"
#include <cstddef>

namespace cpu
{

    dbl_t sigmoid(dbl_t x);

    dbl_t sigmoid_prime(dbl_t x);


    //returns the max value in range [begin, end[
    dbl_t max(const dbl_t* begin, const dbl_t* end);

    //returns the sum of the values in range [begin, end[
    dbl_t sum(const dbl_t* begin, const dbl_t* end);

    /**
     * perform matrix matrix multiplication
     * out = a * b
     * a - matrix (m * n)
     * b - matrix (n * p)
     * out - matrix (m * p)
     */
    void mm_mul(const dbl_t* a, const dbl_t* b, dbl_t* out,
                std::size_t m, std::size_t n, std::size_t p);

    /**
     * perform matrix - vector addition
     * row vector brodcasted and added to all lines of a
     * out = a + b
     * a - matrix (m * n)
     * b - vector (n)
     * out - matrix (m * n)
     */
    void mvrow_add(const dbl_t* a, const dbl_t* b, dbl_t* out,
                   std::size_t m, std::size_t n);

    /**
     * Perform sigmoid operation of a vector
     * out = sigmoid(a)
     * a - vector (n)
     * out - vector (n)
     */
    void vect_sigmoid(const dbl_t* a, dbl_t* out, std::size_t n);

    /**
     * Compute the mean square error between y and y_hat
     * res = mse(y, y_hat)
     * y - matrix (m * n)
     * y_hat - matrix (m * n)
     */
    dbl_t mse(const dbl_t* y, const dbl_t* y_hat, std::size_t m, std::size_t n);

    /**
     * Compute the softmax function
     * out = softmax(a)
     * a - matrix (m * n)
     * out - matrix (m * n)
     */
    void softmax(const dbl_t* a, dbl_t* out, std::size_t m, std::size_t n);

    /**
     * Compute element-wise log of the softmax function
     * out = log(softmax(in))
     * in - matrix (m * n)
     * out - matrix (m * n)
     */
    void log_softmax(const dbl_t* in, dbl_t* out, std::size_t m, std::size_t n);


    /**
     * Cmpute the cross-entrpy cost between y and softmax(logits)
     * res = cross_entropy(y, softmax(logits))
     * y - matrix (m * n) - labels
     * logits - matrix (m * n) - logits of final layer (before y_hat = softmax(logits)
     */
    dbl_t softmax_cross_entropy(const dbl_t* y, const dbl_t* logits,
                                std::size_t m, std::size_t n);

/**
 * Input and kernel are two 4D tensors, out contain
 * the conv2d result between input and kernel
 * - Input is formatted with this form [batch, height, width, in_channels]
 * - Kernel is formatted with this form [height, width, in_channels, out_channels]
 * - Out is formatted with this form [batch, nbBoxH, nbBoxW, nbFilter]
 * - Strides is an array of size 2, the first item is the Y-axis stride,
 *  the other one the X-axis stride
 * - Input_size is an array of size 4 containing the size of the input
 *   tensor with respect of the format given above
 * - Kernel_size is an array of size 4 containing the size of the kernel
 * tensor with respect of the format given above
 */
    void conv2d(const dbl_t* input, const dbl_t* kernel, dbl_t* out,
                const int* strides,
                const int* input_size, const int* kernel_size);


    dbl_t relu(dbl_t x);

    dbl_t relu_prime(dbl_t x);

    dbl_t relu_leaky(dbl_t x, const dbl_t alpha);

    dbl_t relu_leaky_prime(dbl_t x, const dbl_t alpha);

    dbl_t tanh(dbl_t x);

    dbl_t tanh_prime(dbl_t x);

    dbl_t sigmoid_cross_entropy(dbl_t x);

   /**
     * Perform relu operation of a vector
     * out = relu (a)
     * a - vector (n)
     * out - vector (n)
     */

    void vect_relu(const dbl_t* a, dbl_t* out, std::size_t n);

   /**
     * Perform leaky relu operation of a vector
     * out = relu_leaky (a)
     * a - vector (n)
     * out - vector (n)
     */
    void vect_relu_leaky(const dbl_t* a, dbl_t* out, std::size_t n,
                         const dbl_t alpha);

   /**
     * Perform tanh operation of a vector
     * out = tanh (a)
     * a - vector (n)
     * out - vector (n)
     */
    void vect_tanh(const dbl_t* a, dbl_t* out, std::size_t n);

}

#include "ops.hxx"
