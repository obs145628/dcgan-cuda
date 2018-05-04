#pragma once

#include "ops.hh"

#include <cassert>
#include <cmath>

namespace cpu
{


    inline dbl_t sigmoid(dbl_t x)
    {
	//return x;
	return 1.0 / (1.0 + std::exp(-x));
    }


    inline dbl_t sigmoid_prime(dbl_t x)
    {
	return std::exp(-x) / ((1.0 + std::exp(-x) * (1.0 + std::exp(-x))));
    }

    inline dbl_t max(const dbl_t* begin, const dbl_t* end)
    {
	assert(begin != end);
	dbl_t res = *begin;
	while (begin != end)
	{
	    if (*begin > res)
		res = *begin;
	    ++begin;
	}
	return res;
    }

    inline dbl_t sum(const dbl_t* begin, const dbl_t* end)
    {
	dbl_t res = 0;
	while (begin != end)
	    res += *begin++;
	return res;
    }

    inline void mm_mul(const dbl_t* a, const dbl_t* b, dbl_t* out,
		std::size_t m, std::size_t n, std::size_t p)
    {
	for (std::size_t i = 0; i < m; ++i)
	{
	    const dbl_t* ai = a + i * n;
	    for (std::size_t j = 0; j < p; ++j)
	    {
		const dbl_t* bj = b + j;
		dbl_t x = 0;
		for (std::size_t k = 0; k < n; ++k)
		    x += ai[k] * bj[k * p];
		out[i * p + j] = x;
	    }
	}
    }

    inline void mvrow_add(const dbl_t* a, const dbl_t* b, dbl_t* out,
			  std::size_t m, std::size_t n)
    {
	for (std::size_t i = 0; i < m; ++i)
	{
	    const dbl_t* ai = a + i * n;
	    dbl_t* outi = out + i * n;
	    for (std::size_t j = 0; j < n; ++j)
		outi[j] = ai[j] + b[j]; 
	}
    }

    inline void vect_sigmoid(const dbl_t* a, dbl_t* out, std::size_t n)
    {
	for (std::size_t i = 0; i < n; ++i)
	    out[i] = sigmoid(a[i]);
    }

    inline dbl_t mse(const dbl_t* y, const dbl_t* y_hat, std::size_t m, std::size_t n)
    {
	std::size_t len = m * n;
	dbl_t res = 0;
	for (std::size_t i = 0; i < len; ++i)
	    res += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
	return res / len;
    }


    inline void softmax(const dbl_t* a, dbl_t* out, std::size_t m, std::size_t n)
    {
	for (std::size_t i = 0; i < m; ++i)
	{
	    dbl_t max_input = max(a + i * n, a + (i + 1) * n);

	    for (std::size_t j = 0; j < n; ++j)
		out[i * n + j] = std::exp(a[i * n + j] - max_input);

	    dbl_t sum_ex = sum(out + i * n, out + (i + 1) * n);

	    for (std::size_t j = 0; j < n; ++j)
		out[i * n + j] = out[i * n + j] / sum_ex;
	}
    }

    inline void log_softmax(const dbl_t* in, dbl_t* out, std::size_t m, std::size_t n)
    {
	for (std::size_t i = 0; i < m; ++i)
	{
	    dbl_t max_input = max(in + i * n, in + (i + 1) * n);

	    dbl_t e_x = 0;
	    for (std::size_t j = 0; j < n; ++j)
		e_x += std::exp(in[i * n + j] - max_input);
	    dbl_t logsum = max_input + std::log(e_x);

	    for (std::size_t j = 0; j < n; ++j)
		out[i * n + j] = in[i * n + j] - logsum;
	}
    }

    inline dbl_t softmax_cross_entropy(const dbl_t* y, const dbl_t* logits,
				       std::size_t m, std::size_t n)
    {
	dbl_t res = 0;

	for (std::size_t i = 0; i < m; ++i)
	{

	    dbl_t max_logits = max(logits + i * n, logits + (i + 1) * n);

	    dbl_t e_x = 0;
	    for (std::size_t j = 0; j < n; ++j)
		e_x += std::exp(logits[i * n + j] - max_logits);
	    dbl_t logsum = max_logits + std::log(e_x);

	    for (std::size_t j = 0; j < n; ++j)
		res += y[i * n + j] * (logits[i * n + j] - logsum);
	}
    
	return - res / m;
    }


}
