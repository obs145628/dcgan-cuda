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

    inline void tmm_mul(const dbl_t* a, const dbl_t* b, dbl_t* out,
                        std::size_t m, std::size_t n, std::size_t p)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            const dbl_t* ai = a + i;
            for (std::size_t j = 0; j < p; ++j)
            {
                const dbl_t* bj = b + j;
                dbl_t x = 0;
                for (std::size_t k = 0; k < n; ++k)
                    x += ai[k * m] * bj[k * p];
                out[i * p + j] = x;
            }
        }
    }

    inline void mtm_mul(const dbl_t* a, const dbl_t* b, dbl_t* out,
                       std::size_t m, std::size_t n, std::size_t p)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            const dbl_t* ai = a + i * n;
            for (std::size_t j = 0; j < p; ++j)
            {
                const dbl_t* bj = b + j * n;
                dbl_t x = 0;
                for (std::size_t k = 0; k < n; ++k)
                    x += ai[k] * bj[k];
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

    inline void conv2d(const dbl_t* input, const dbl_t* kernel, dbl_t* out,
                       const int* strides,
                       const int* input_size, const int* kernel_size)
    {
        const std::size_t kernelH = kernel_size[0];
        const std::size_t kernelW = kernel_size[1];
        const std::size_t nbFilter = kernel_size[3];

        const std::size_t nbImage = input_size[0];
        const std::size_t nbChan = input_size[3];
        const std::size_t inputH = input_size[1];
        const std::size_t inputW = input_size[2];

        const std::size_t stepLenH = (inputH - kernelH) / strides[0] + 1;
        const std::size_t stepLenW = (inputW - kernelW) / strides[1] + 1;

        for (std::size_t b = 0; b < nbImage; ++b)
            for (std::size_t i = 0; i < stepLenH; ++i)
                for (std::size_t j = 0; j < stepLenW; ++j)
                    for (std::size_t k = 0; k < nbFilter; ++k)
                    {
                        dbl_t val = 0;
                        for (std::size_t q = 0; q < nbChan; ++q)
                            for (std::size_t di = 0; di < kernelH; ++di)
                                for (std::size_t dj = 0; dj < kernelW; ++dj)
                                {
                                    std::size_t imgIndex = b * inputH * inputW * nbChan;
                                    std::size_t ihIndex = (strides[0] * i + di) * inputW * nbChan;
                                    std::size_t iwIndex = (strides[1] * j + dj) * nbChan;

                                    std::size_t khIndex = di * kernelW * nbChan * nbFilter;
                                    std::size_t kwIndex = dj * nbChan * nbFilter;
                                    std::size_t chIndex = q * nbFilter;
                                    val += input[imgIndex + ihIndex + iwIndex + q]
                                        * kernel[khIndex + kwIndex + chIndex + k];
                                }
                        std::size_t oimgIndex = b * stepLenH * stepLenW * nbFilter;
                        std::size_t ohIndex = i * stepLenW * nbFilter;
                        std::size_t owIndex = j * nbFilter;
                        out[oimgIndex + ohIndex + owIndex + k] = val;
                    }
    }
    
    inline void conv2d_bias_add(const dbl_t* z, const dbl_t* bias, dbl_t* out,
                                const int* input_size)
    {
        const std::size_t batch = input_size[0];
        const std::size_t height = input_size[1];
        const std::size_t width = input_size[2];
        const std::size_t outputCh = input_size[3];
        
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t i = 0; i < height; ++i)
                for (std::size_t j = 0; j < width; ++j)
                    for (std::size_t k = 0; k < outputCh; ++k)
                    {
                        std::size_t imgIndex = b * height * width * outputCh;
                        std::size_t hIndex = i * width * outputCh;
                        std::size_t wIndex = j * outputCh;
                        out[imgIndex + hIndex + wIndex + k] = z[imgIndex + hIndex + wIndex + k] + bias[k];
                    }
    }

    inline dbl_t relu(dbl_t x)
    {
        return x < 0 ? 0 : x;
    }

    inline dbl_t relu_prime(dbl_t x)
    {
        return x < 0 ? 0 : 1;
    }

    inline void vect_relu(const dbl_t* a, dbl_t* out, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = relu(a[i]);
    }

    inline dbl_t relu_leaky(dbl_t x, const dbl_t alpha)
    {
        return x < 0 ? alpha * x : x;
    }

    inline dbl_t relu_leaky_prime(dbl_t x, const dbl_t alpha = 0.2)
    {
        return x < 0 ? alpha : 1;
    }

    inline void vect_relu_leaky(const dbl_t* a, dbl_t* out, std::size_t n,
                                const dbl_t alpha)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = relu_leaky(a[i], alpha);
    }

    inline dbl_t tanh(dbl_t x)
    {
        return (2.0 / (1.0 + std::exp(-2 * x))) - 1.0;
    }

    inline dbl_t tanh_prime(dbl_t x)
    {
        return 1.0 - tanh(x) * tanh(x);
    }

    inline void vect_tanh(const dbl_t* a, dbl_t* out, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = tanh(a[i]);
    }

    inline void vect_sub_coeff(const dbl_t* a, const dbl_t* b, dbl_t coeff, dbl_t* out, 
                               std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = coeff * (a[i] - b[i]);
    }

    inline void sigmoid_grad(const dbl_t* sig_out, const dbl_t* dout, dbl_t* out,
                             std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = sig_out[i] * (1 - sig_out[i]) * dout[i];
    }

    inline void mat_mul_add(const dbl_t* x, const dbl_t* w, const dbl_t* b, dbl_t* out,
                            std::size_t m, std::size_t n, std::size_t p)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            const dbl_t* xi = x + i * n;
            for (std::size_t j = 0; j < p; ++j)
            {
                const dbl_t* wj = w + j;
                dbl_t x = 0;
                for (std::size_t k = 0; k < n; ++k)
                    x += xi[k] * wj[k * p];
                out[i * p + j] = x + b[j];
            }
        }
    }

    inline void mat_sum_rows(const dbl_t* a, dbl_t* out,
                             std::size_t m, std::size_t n)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            const dbl_t* ai = a + i * n;
            dbl_t sum = 0;
            for (std::size_t j = 0; j < n; ++j)
                sum += ai[j];
            out[i] = sum;
        }
    }

    inline void mat_sum_cols(const dbl_t* a, dbl_t* out,
                             std::size_t m, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            const dbl_t* ai = a + i;
            dbl_t sum = 0;
            for (std::size_t j = 0; j < m; ++j)
                sum += ai[j * n];
            out[i] = sum;
        }
    }

    inline void softmax_cross_entropy_grad(const dbl_t* y, const dbl_t* logits, dbl_t* out,
                                           std::size_t m, std::size_t n)
    {
        softmax(logits, out, m, n);
        for (std::size_t i = 0; i < m * n; ++i)
            out[i] = (out[i] - y[i]) / m;
    }

    inline void relu_grad(const dbl_t* z, const dbl_t* dout, dbl_t* out,
                          std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = z[i] > 0 ? dout[i] : 0;
    }

}
