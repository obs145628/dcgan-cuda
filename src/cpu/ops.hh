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
     * Perform segmoid operation of a vector
     * out = sigmoid(a)
     * a - vector (n)
     * out-  vector (n)
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
    
}

#include "ops.hxx"
