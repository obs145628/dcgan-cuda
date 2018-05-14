#pragma once

#include "ops.hh"

#include <cassert>
#include <cmath>
#include <stdlib.h>
#include <algorithm>

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

    inline std::size_t argmax(const dbl_t* begin, const dbl_t* end)
    {
        std::size_t res = 0;
        std::size_t len = end - begin;
        for (std::size_t i = 1; i < len; ++i)
            if (begin[i] > begin[res])
                res = i;
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
                       const int* strides, int pad_height, int pad_width,
                       FilterAccessor* faI, FilterAccessor* faK, int valid)
    {
        const std::size_t kernelH = faK->size_get(0);
        const std::size_t kernelW = faK->size_get(1);
        const std::size_t nbFilter = faK->size_get(3);

        const std::size_t nbImage = faI->size_get(0);
        const std::size_t nbChan =  faI->size_get(3);
        const std::size_t inputH =  faI->size_get(1);
        const std::size_t inputW =  faI->size_get(2);

        std::size_t stepLenH = 0;
        std::size_t stepLenW = 0;
        const int pad_top = pad_height / 2;
        const int pad_left = pad_width / 2;
        if (valid)
        {
            stepLenH = (inputH + pad_height - kernelH) / strides[0] + 1;
            stepLenW = (inputW + pad_width - kernelW) / strides[1] + 1;
        }
        else
        {
            stepLenH = (std::size_t)std::ceil(
                        static_cast<float>(inputH) / (float)strides[0]);
            stepLenW = (std::size_t)std::ceil(
                        static_cast<float>(inputW) / (float)strides[1]);
        }

        for (std::size_t b = 0; b < nbImage; ++b)
        {
            for (std::size_t i = 0; i < stepLenH; ++i)
            {
                for (std::size_t j = 0; j < stepLenW; ++j)
                {
                    for (std::size_t k = 0; k < nbFilter; ++k)
                    {
                        dbl_t val = 0;
                        for (std::size_t q = 0; q < nbChan; ++q)
                        {
                            for (std::size_t di = 0; di < kernelH; ++di)
                            {
                                for (std::size_t dj = 0; dj < kernelW; ++dj)
                                {
                                  int hIndex = strides[0] * i + di - pad_top;
                                  int wIndex = strides[1] * j + dj - pad_left;
                                  if (hIndex >= 0 && wIndex >= 0
                                      && (size_t)hIndex < inputH
                                      && (size_t)wIndex < inputW)
                                  {
                                    int inputInd = faI->access(b, hIndex, wIndex, q);
                                    int kernelInd = faK->access(di, dj, q, k);
                                    val += input[inputInd] * kernel[kernelInd];
                                  }
                                }
                            }
                        }
                        std::size_t oimgIndex = b * stepLenH * stepLenW * nbFilter;
                        std::size_t ohIndex = i * stepLenW * nbFilter;
                        std::size_t owIndex = j * nbFilter;
                        out[oimgIndex + ohIndex + owIndex + k] = val;
                    }
                }
            }
        }
    }

    inline void conv2d(const dbl_t* input, const dbl_t* kernel, dbl_t* out,
                       const int* strides, int pad_height, int pad_width,
                       const int* input_size, const int* kernel_size,
                       int valid = 0)
    {
        IdentityAccessor* iaI = new IdentityAccessor(input_size);
        IdentityAccessor* iaK = new IdentityAccessor(kernel_size);
        conv2d(input, kernel, out, strides, pad_height, pad_width, iaI, iaK, valid);
        delete iaI;
        delete iaK;
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

    inline void conv2d_bias_add_grad(const dbl_t* z, const int* size, dbl_t* out)
    {
        const std::size_t batch = size[0];
        const std::size_t height = size[1];
        const std::size_t width = size[2];
        const std::size_t outputCh = size[3];

        for (std::size_t k = 0; k < outputCh; ++k)
            out[k] = 0;

        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t i = 0; i < height; ++i)
                for (std::size_t j = 0; j < width; ++j)
                    for (std::size_t k = 0; k < outputCh; ++k)
                    {
                        std::size_t imgIndex = b * height * width * outputCh;
                        std::size_t hIndex = i * width * outputCh;
                        std::size_t wIndex = j * outputCh;

                        out[k] += z[imgIndex + hIndex + wIndex + k];
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

    inline void vect_update(const dbl_t* dv, dbl_t* out, dbl_t coeff,
                            std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] += coeff * dv[i];
    }

    inline dbl_t sigmoid_cross_entropy(const dbl_t* y, const dbl_t* x,
                                       std::size_t n)
    {
        dbl_t res = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
             if (x[i] >= 0)
                 res += x[i] - x[i] * y[i] + std::log(1 + std::exp(-x[i]));
             else
                 res += - x[i] * y[i] + std::log(1 + std::exp(x[i]));
        }
        return res / n;
    }

    inline void sigmoid_cross_entropy_grad(const dbl_t* y, const dbl_t* logits, dbl_t* out,
                                           std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = (sigmoid(logits[i]) - y[i]) / n;
    }

    inline void tanh_grad(const dbl_t* tanh_out, const dbl_t* dout, dbl_t* out,
                             std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = (1 - std::pow(tanh_out[i], 2.0)) * dout[i];
    }

    inline std::size_t argmax_acc(const dbl_t* y, const dbl_t* y_hat,
                                  std::size_t m, std::size_t n)
    {
        std::size_t res = 0;
        for (std::size_t i = 0; i < m; ++i)
            res += argmax(y + i * n, y + (i + 1) * n)
                == argmax(y_hat + i * n, y_hat + (i + 1) * n);
        return res;
    }

    inline void padd_full_conv(const dbl_t* input, dbl_t* out, int stride,
                               const int* out_size,
                               FilterAccessor* faI)
    {
          const int sOutTot0 = out_size[1] * out_size[2] * out_size[3];
          const int sOutTot1 = out_size[2] * out_size[3];
          for (int img = 0; img < out_size[0]; ++img)
              for (int hIndex = out_size[4]; hIndex < out_size[1]; hIndex += stride)
                  for (int wIndex = out_size[5]; wIndex < out_size[2]; wIndex += stride)
                      for (int chan = 0; chan < out_size[3]; ++chan)
                      {
                          int prevHIndex = hIndex - out_size[4] - ((hIndex - out_size[4]) / stride * (stride - 1));
                          int prevWIndex = wIndex - out_size[5] - ((wIndex - out_size[5]) / stride * (stride - 1));
                          if (prevHIndex >= 0 && prevHIndex < faI->size_get(1)
                              && prevWIndex >= 0 && prevWIndex < faI->size_get(2))
                          {
                              int inputInd = faI->access(img, prevHIndex, prevWIndex, chan);
                              out[img * sOutTot0 + hIndex * sOutTot1
                                  + wIndex * out_size[3] + chan] = input[inputInd];
                          }
                      }
    }

    inline void padd_ker(const dbl_t* kernel, dbl_t* out, int stride,
                         const int* out_size, FilterAccessor* fa)
    {
        const int sOutTot0 = out_size[1] * out_size[2] * out_size[3];
        const int sOutTot1 = out_size[2] * out_size[3];

        for (int hIndex = 0; hIndex < out_size[0]; hIndex += stride)
            for (int wIndex = 0; wIndex < out_size[1]; wIndex += stride)
                for (int inCh = 0; inCh < out_size[2]; ++inCh)
                    for (int oCh = 0; oCh < out_size[3]; ++oCh)
                    {
                        int prevHIndex = hIndex - (hIndex / stride * (stride - 1));
                        int prevWIndex = wIndex - (wIndex / stride * (stride - 1));
                        int kernelInd = fa->access(prevHIndex, prevWIndex, inCh, oCh);
                        out[hIndex * sOutTot0 + wIndex * sOutTot1
                            + inCh * out_size[3] + oCh] = kernel[kernelInd];
                    }
    }

    inline dbl_t* formatDw(dbl_t* dldw, const int* size)
    {
        dbl_t* out = (dbl_t*)malloc(size[0] * size[1] * size[2] * size[3] * sizeof(dbl_t));
        const int sOutTot0 = size[2] * size[0] * size[3];
        const int sOutTot1 = size[0] * size[3];

        const int sdlTot0 = size[1] * size[2] * size[3];
        const int sdlTot1 = size[2] * size[3];

        for (int hIndex = 0; hIndex < size[1]; ++hIndex)
            for (int wIndex = 0; wIndex < size[2]; ++wIndex)
                for (int ch = 0; ch < size[0]; ++ch)
                    for (int nbFilter = 0; nbFilter < size[3]; ++nbFilter)
                        out[hIndex * sOutTot0 + wIndex * sOutTot1
                            + ch * size[3] + nbFilter] = dldw[ch * sdlTot0
                                                              + hIndex * sdlTot1
                                                              + wIndex * size[3] + nbFilter];

        free(dldw);
        return out;
    }

    inline void tensor_add(dbl_t* t1, dbl_t* t2, const int* size)
    {
        int sTot = size[0] * size[1] * size[2] * size[3];
        for (int i = 0; i < sTot; ++i)
            t1[i] = t1[i] + t2[i];
    }

    inline dbl_t* tensor_concat_axis3(dbl_t* t1, dbl_t* t2, const int* size_t1, const int* size_t2)
    {
        dbl_t* out = (dbl_t*)malloc(size_t1[0] * size_t1[1] * size_t1[2]
                                      * (size_t1[3] + size_t2[3]) * sizeof(dbl_t));
        const int new_axis3_size = (size_t1[3] + size_t2[3]);
        const int sOutTot0 = size_t1[1] * size_t1[2] * (new_axis3_size);
        const int sOutTot1 = size_t1[2] * new_axis3_size;

        const int st1Tot0 = size_t1[1] * size_t1[2] * size_t1[3];
        const int st1Tot1 = size_t1[2] * size_t1[3];

        const int st2Tot0 = size_t2[1] * size_t2[2] * size_t2[3];
        const int st2Tot1 = size_t2[2] * size_t2[3];

        for (int hIndex = 0; hIndex < size_t1[0]; ++hIndex)
            for (int wIndex = 0; wIndex < size_t1[1]; ++wIndex)
                for (int ch = 0; ch < size_t1[2]; ++ch)
                    for (int nbFilter = 0; nbFilter < new_axis3_size; ++nbFilter)
                    {
                      if (nbFilter >= size_t1[3])
                          out[hIndex * sOutTot0 + wIndex * sOutTot1
                              + ch * new_axis3_size + nbFilter] = t2[hIndex * st2Tot0
                                                  + wIndex * st2Tot1 + ch * size_t2[3]
                                                  + nbFilter - size_t1[3]];
                      else
                          out[hIndex * sOutTot0 + wIndex * sOutTot1
                              + ch * new_axis3_size + nbFilter] = t1[hIndex * st1Tot0
                                                  + wIndex * st1Tot1 + ch * size_t1[3]
                                                  + nbFilter];
                    }
        free(t1);
        free(t2);
        return out;
    }

    inline dbl_t* tensor_concat_axis0(dbl_t* t1, dbl_t* t2, const int* size_t1, const int* size_t2)
    {
        const int new_size = size_t1[0] * size_t1[1] * size_t1[2] * size_t1[3]
                       + size_t2[0] * size_t2[1] * size_t2[2] * size_t2[3];
        dbl_t* out = (dbl_t*)malloc(new_size * sizeof(dbl_t));
        memcpy(out, t1, size_t1[0] * size_t1[1] * size_t1[2] * size_t1[3] * sizeof(dbl_t));
        memcpy(out + size_t1[0] * size_t1[1] * size_t1[2] * size_t1[3], t2,
              size_t2[0] * size_t2[1] * size_t2[2] * size_t2[3] * sizeof(dbl_t));
        free(t1);
        free(t2);
        return out;
    }

    inline void conv2d_input_grad(const dbl_t* dX1, const dbl_t* W1, const int stride, const int* dX1_size,
                                  const int* W1_size, dbl_t* out, const int* input_size)
    {
      const int nbChan = W1_size[2];
      const int nbFilter = dX1_size[3];

      dbl_t* dLdX = nullptr;
      int add_size[4] = {0,0,0,0};
      int concat_size[4] = {0,0,0,0};

      auto padded_filter_tab = new padded_filter[nbFilter];

      for (int filter = 0; filter < nbFilter; ++filter)
      {
          YFilterAccessor* yf = new YFilterAccessor(dX1_size, filter);

          const int striddedWidth = (yf->size_get(2) - 1) * (stride - 1) + yf->size_get(2);
          const int striddedHeight = (yf->size_get(1) - 1) * (stride - 1) + yf->size_get(1);
          const int pad_h = input_size[0] - (striddedHeight - W1_size[0] + 1);
          const int pad_w = input_size[1] - (striddedWidth - W1_size[1] + 1);
          const int outHeight = striddedHeight + pad_h;
          const int outWidth = striddedWidth + pad_w;
          dbl_t* padded = (dbl_t*)calloc(yf->size_get(0) * outHeight
                                           * outWidth * yf->size_get(3),
                                           sizeof(dbl_t));
          const int pad_bottom = pad_h / 2;
          const int pad_top = pad_h - pad_bottom;
          const int pad_right = pad_w / 2;
          const int pad_left = pad_w - pad_right;

          const int padded_size[6] =
          {
              yf->size_get(0), outHeight, outWidth, yf->size_get(3),
              pad_top, pad_left
          };

          padd_full_conv(dX1, padded, stride, padded_size, yf);
          padded_filter_tab[filter].data = padded;
          padded_filter_tab[filter].size[0] = padded_size[0];
          padded_filter_tab[filter].size[1] = padded_size[1];
          padded_filter_tab[filter].size[2] = padded_size[2];
          padded_filter_tab[filter].size[3] = padded_size[3];
          padded_filter_tab[filter].size[4] = padded_size[4];
          padded_filter_tab[filter].size[5] = padded_size[5];
          padded_filter_tab[filter].acc = yf;
      }

      for (int chan = 0; chan < nbChan; ++chan)
      {
          dbl_t* dLdXFilter = nullptr;
          for (int filter = 0; filter < nbFilter; ++filter)
          {
              WFilterRot180Accessor* wf = new WFilterRot180Accessor(W1_size, filter, chan);

              auto padded_filter_cur = padded_filter_tab[filter];

              const int stepLenH = (padded_filter_cur.size[1] - wf->size_get(0)) + 1;
              const int stepLenW = (padded_filter_cur.size[2] - wf->size_get(1)) + 1;
              dbl_t* out_conv = (dbl_t*)calloc(padded_filter_cur.size[0] * stepLenH
                                                * stepLenW * wf->size_get(3), sizeof(dbl_t));
              const int strides[2] = {1, 1};
              IdentityAccessor* ia = new IdentityAccessor(padded_filter_cur.size);
              conv2d(padded_filter_cur.data, W1, out_conv, strides, 0, 0, ia, wf, 1);

              add_size[0] = padded_filter_cur.size[0];
              add_size[1] = stepLenH;
              add_size[2] = stepLenW;
              add_size[3] = wf->size_get(3);

              delete wf;
              delete ia;
              if (dLdXFilter == nullptr)
                  dLdXFilter = out_conv;
              else
              {
                  tensor_add(dLdXFilter, out_conv, add_size);
                  free(out_conv);
              }
          }
          if (dLdX == nullptr)
          {
              dLdX = dLdXFilter;
              concat_size[0] = add_size[0];
              concat_size[1] = add_size[1];
              concat_size[2] = add_size[2];
              concat_size[3] = add_size[3];
          }
          else
          {
            dLdX = tensor_concat_axis3(dLdX, dLdXFilter, concat_size, add_size);
            concat_size[3] += add_size[3];
          }
      }
      memcpy(out, dLdX, concat_size[0] * concat_size[1] * concat_size[2] * concat_size[3] * sizeof(dbl_t));
      free(dLdX);
      for (int filter = 0; filter < nbFilter; ++filter)
      {
          free(padded_filter_tab[filter].data);
          delete padded_filter_tab[filter].acc;
      }
      delete[] padded_filter_tab;
    }

    inline void conv2d_kernel_grad(const dbl_t* dX1, const dbl_t* X0, const int stride, const int* dX1_size, const int* X0_size,
                                   dbl_t* out, const int* padded_size_input)
    {
        const int nbFilterTotal = dX1_size[3];
        const int nbChan = X0_size[3];
        const int nbImg = X0_size[0];
        dbl_t* dLdW = nullptr;
        int add_size[4] = {0,0,0,0};
        int concat_size[4] = {0,0,0,0};

        auto padded_img_tab = new padded_img[nbImg];

        for (int img = 0; img < nbImg; ++img)
        {
            YtoKerAccessor* yk = new YtoKerAccessor(dX1_size, nbFilterTotal, img);

            const int striddedWidth = (yk->size_get(1) - 1) * (stride - 1) + yk->size_get(1);
            const int striddedHeight = (yk->size_get(0) - 1) * (stride - 1) + yk->size_get(0);
            dbl_t* padded = (dbl_t*)calloc(striddedHeight * striddedWidth
                                           * yk->size_get(2) * yk->size_get(3)
                                           , sizeof(dbl_t));
            const int padded_size[4] =
            {
                striddedHeight, striddedWidth, yk->size_get(2), yk->size_get(3)
            };

            padd_ker(dX1, padded, stride, padded_size, yk);
            padded_img_tab[img].img = img;
            padded_img_tab[img].data = padded;
            padded_img_tab[img].size[0] = striddedHeight;
            padded_img_tab[img].size[1] = striddedWidth;
            padded_img_tab[img].size[2] = padded_size[2];
            padded_img_tab[img].size[3] = padded_size[3];
            padded_img_tab[img].acc = yk;
        }


        for (int chan = 0; chan < nbChan; ++chan)
        {
            dbl_t* dLdWImg = nullptr;
            for (int img = 0; img < nbImg; ++img)
            {
                ChFilterAccessor* ch = new ChFilterAccessor(X0_size, img, chan, 0, 0);

                auto padded_img_cur = padded_img_tab[img];

                const int stepLenH = (ch->size_get(1) + padded_size_input[0]
                                      - padded_img_cur.size[0]) + 1;
                const int stepLenW = (ch->size_get(2) + padded_size_input[1]
                                      - padded_img_cur.size[1]) + 1;

                dbl_t* out_conv = (dbl_t*)calloc(ch->size_get(0) * stepLenH
                                                  * stepLenW
                                                  * padded_img_cur.size[3],
                                                  sizeof(dbl_t));
                const int strides[2] = {1, 1};

                IdentityAccessor* ia = new IdentityAccessor(padded_img_cur.size);
                conv2d(X0, padded_img_cur.data, out_conv, strides, padded_size_input[0],
                       padded_size_input[1], ch, ia, 1);

                add_size[0] = ch->size_get(0);
                add_size[1] = stepLenH;
                add_size[2] = stepLenW;
                add_size[3] = padded_img_cur.size[3];

                delete ch;
                delete ia;
                if (dLdWImg == nullptr)
                    dLdWImg = out_conv;
                else
                {
                    tensor_add(dLdWImg, out_conv, add_size);
                    free(out_conv);
                }
            }
            if (dLdW == nullptr)
            {
                dLdW = dLdWImg;
                concat_size[0] = add_size[0];
                concat_size[1] = add_size[1];
                concat_size[2] = add_size[2];
                concat_size[3] = add_size[3];
            }
            else
            {
              dLdW = tensor_concat_axis0(dLdW, dLdWImg, concat_size, add_size);
              concat_size[0] += add_size[0];
            }
        }
        dLdW = formatDw(dLdW, concat_size);
        memcpy(out, dLdW, concat_size[0] * concat_size[1] * concat_size[2] * concat_size[3] * sizeof(dbl_t));
        free(dLdW);
        for (int img = 0; img < nbImg; ++img)
        {
            free(padded_img_tab[img].data);
            delete padded_img_tab[img].acc;
        }
        delete[] padded_img_tab;
    }

    inline void conv2d_transpose(const dbl_t* input, const dbl_t* kernel, const int* out_size, const int stride,
                                 dbl_t* out, const int* input_size, const int* kernel_size)
    {
        int out_size_grad[2] = {out_size[1], out_size[2]};
        conv2d_input_grad(input, kernel, stride, input_size, kernel_size, out, out_size_grad);
    }

    inline void conv2d_transpose_input_grad(const dbl_t* dX1, const dbl_t* W1, const int stride, const int* dX1_size,
                                            const int* W1_size, dbl_t* out, const int* input_size)
    {
      IdentityAccessor* ia1 = new IdentityAccessor(dX1_size);
      IdentityAccessor* ia2 = new IdentityAccessor(W1_size);

      (void)input_size;
      const int strides[2] = {stride, stride};
      conv2d(dX1, W1, out, strides, 0, 0, ia1, ia2, 1);

      delete ia1;
      delete ia2;
    }

    inline void conv2d_transpose_kernel_grad(const dbl_t* dX1, const dbl_t* X0, const int stride, const int* dX1_size, const int* X0_size,
                                             dbl_t* out)
    {
        const int nbChan = dX1_size[3];
        const int nbImg = X0_size[0];
        dbl_t* dLdW = nullptr;
        int add_size[4] = {0,0,0,0};
        int concat_size[4] = {0,0,0,0};


        for (int chan = 0; chan < nbChan; ++chan)
        {
            dbl_t* dLdWImg = nullptr;
            for (int img = 0; img < nbImg; ++img)
            {

                ChFilterAccessor* ch = new ChFilterAccessor(dX1_size, img, chan, 0, 0);
                YtoKerTransposeAccessor* yk = new YtoKerTransposeAccessor(X0_size, chan, img);

                const int striddedWidth = (yk->size_get(1) - 1) * (stride - 1) + yk->size_get(1);
                const int striddedHeight = (yk->size_get(0) - 1) * (stride - 1) + yk->size_get(0);
                dbl_t* padded = (dbl_t*)calloc(striddedHeight * striddedWidth
                                           * yk->size_get(2) * yk->size_get(3)
                                           , sizeof(dbl_t));
                const int padded_size[4] =
                {
                    striddedHeight, striddedWidth, yk->size_get(2), yk->size_get(3)
                };

                padd_ker(X0, padded, stride, padded_size, yk);

                const int stepLenH = (ch->size_get(1)
                                      - padded_size[0]) + 1;
                const int stepLenW = (ch->size_get(2)
                                      - padded_size[1]) + 1;

                dbl_t* out_conv = (dbl_t*)calloc(ch->size_get(0) * stepLenH
                                                  * stepLenW
                                                  * padded_size[3],
                                                  sizeof(dbl_t));
                const int strides[2] = {1, 1};

                //TransposeKerAccessor* tka = new TransposeKerAccessor(padded_size);
                IdentityAccessor* ia = new IdentityAccessor(padded_size);
                conv2d(dX1, padded, out_conv, strides, 0, 0, ch, ia, 1);

                add_size[0] = ch->size_get(0);
                add_size[1] = stepLenH;
                add_size[2] = stepLenW;
                add_size[3] = padded_size[3];

                delete ch;
                delete ia;
                delete yk;
                free(padded);
                if (dLdWImg == nullptr)
                    dLdWImg = out_conv;
                else
                {
                    tensor_add(dLdWImg, out_conv, add_size);
                    free(out_conv);
                }
            }
            if (dLdW == nullptr)
            {
                dLdW = dLdWImg;
                concat_size[0] = add_size[0];
                concat_size[1] = add_size[1];
                concat_size[2] = add_size[2];
                concat_size[3] = add_size[3];
            }
            else
            {
              dLdW = tensor_concat_axis0(dLdW, dLdWImg, concat_size, add_size);
              concat_size[0] += add_size[0];
            }
        }
        dLdW = formatDw(dLdW, concat_size);
        memcpy(out, dLdW, concat_size[0] * concat_size[1] * concat_size[2] * concat_size[3] * sizeof(dbl_t));
        free(dLdW);
    }


    inline void moment_update(const dbl_t* dv, dbl_t* out,
                              dbl_t a, dbl_t b, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            out[i] = a * out[i] + b * dv[i];
    }

    inline void moment_update2(const dbl_t* dv, dbl_t* out,
                               dbl_t a, dbl_t b, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i)
            out[i] = a * out[i] + b * dv[i] * dv[i];
    }

    inline void adam_update(const dbl_t* m, const dbl_t* v, dbl_t* out,
                            dbl_t lrt, dbl_t eps, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = out[i] - lrt * m[i] / (std::sqrt(v[i]) + eps);
    }

    inline void leaky_relu_grad(const dbl_t* z, const dbl_t* dout, dbl_t* out,
                                dbl_t alpha, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = z[i] < 0 ? alpha * dout[i] : dout[i];
    }

}
