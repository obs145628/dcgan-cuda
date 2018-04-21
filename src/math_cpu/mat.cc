#include "mat.hh"
#include <iostream>

void mm_mul(const dbl_t* a, const dbl_t* b, dbl_t* out,
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

void mvrow_add(const dbl_t* a, const dbl_t* b, dbl_t* out,
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


void mat_print(const dbl_t* ptr, std::size_t m, std::size_t n)
{
    for (std::size_t i = 0; i < m; ++i)
    {
	const dbl_t* ptri = ptr + i * n;
	std::cout << "[";
	for (std::size_t j = 0; j < n; ++j)
	{
	    std::cout << ptri[j] << " ";
	}
	std::cout << "]\n";
    }
}
