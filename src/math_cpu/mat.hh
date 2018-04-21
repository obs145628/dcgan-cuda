#pragma once

#include <cstddef>
#include "../config/types.hh"

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

void mat_print(const dbl_t* ptr, std::size_t m, std::size_t n);
