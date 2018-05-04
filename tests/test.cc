#include <iostream>
#include <cmath>
#include "../src/memory/types.hh"

void print(dbl_t* data, std::size_t rows, std::size_t cols)
{
    for (std::size_t i = 0; i < rows; ++i)
    {
	for (std::size_t j = 0; j < cols; ++j)
	    std::cout << data[i * cols + j] << "| ";
	std::cout << std::endl;
    }
}

dbl_t max(dbl_t* begin, dbl_t* end)
{
    dbl_t res = *begin;
    while (begin != end)
    {
	if (*begin > res)
	    res = *begin;
	++begin;
    }
    return res;
}

dbl_t sum(dbl_t* begin, dbl_t* end)
{
    dbl_t res = 0;
    while (begin != end)
	res += *begin++;
    return res;
}

void softmax(dbl_t* in, dbl_t* out, std::size_t rows, std::size_t cols)
{
    for (std::size_t i = 0; i < rows; ++i)
    {
	dbl_t max_input = max(in + i * cols, in + (i + 1) * cols);

	for (std::size_t j = 0; j < cols; ++j)
	    out[i * cols + j] = std::exp(in[i * cols + j] - max_input);

	dbl_t sum_ex = sum(out + i * cols, out + (i + 1) * cols);

	for (std::size_t j = 0; j < cols; ++j)
	    out[i * cols + j] = out[i * cols + j] / sum_ex;
    }
}


dbl_t softmax_cross_entropy_loss(dbl_t* logits, dbl_t* y,
				 std::size_t rows, std::size_t cols)
{
    dbl_t* y_hat = new dbl_t[rows * cols];
    softmax(logits, y_hat, rows, cols);

    dbl_t res = 0;
    for (std::size_t i = 0; i < rows * cols; ++i)
	res += y[i] * std::log(y_hat[i]);
    res = - res / rows;

    delete[] y_hat;
    return res;
}

int main()
{

    dbl_t logits[] = {
	0.1, 1.2, 4.3,
	4.1, 0.2, 7.3,
	0.06, 2.01, 0.23,
	5.6, 2.3, 1.18
    };

    dbl_t y[] = {
	0.1, 0.2, 0.7,
	0.8, .1, .1,
	0.1, 0.3, 0.6,
	.6, .2, .2
    };

    //print(logits, 4, 3);

    //softmax(logits, out, 4, 3);
    //print(out, 4, 3);

    dbl_t loss = softmax_cross_entropy_loss(logits, y, 4, 3);
    print(&loss, 1, 1);
}