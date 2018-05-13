#pragma once

#include "../memory/types.hh"
#include <cstddef>
#include <string.h>

namespace cpu
{

    dbl_t sigmoid(dbl_t x);

    dbl_t sigmoid_prime(dbl_t x);


    //returns the max value in range [begin, end[
    dbl_t max(const dbl_t* begin, const dbl_t* end);

    //compute the index of the maximum value
    std::size_t argmax(const dbl_t* begin, const dbl_t* end);

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
     * perform matrix matrix multiplication
     * out = tr(a) * b
     * a - matrix (n * m)
     * b - matrix (n * p)
     * out - matrix (m * p)
     */
    void tmm_mul(const dbl_t* a, const dbl_t* b, dbl_t* out,
                 std::size_t m, std::size_t n, std::size_t p);

    /**
     * perform matrix matrix multiplication
     * out = a * tr(b)
     * a - matrix (m * n)
     * b - matrix (p * n)
     * out - matrix (m * p)
     */
    void mtm_mul(const dbl_t* a, const dbl_t* b, dbl_t* out,
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
     * Classes used to filter data access from a dblt_t array
     */
    class FilterAccessor
    {
    public:
        FilterAccessor(const int* size)
                : _size(size)
        {
          _stot0 = size[1] * size[2] * size[3];
          _stot1 = size[2] * size[3];
        }

        virtual ~FilterAccessor() = default;

        virtual int access(int ind0, int ind1, int ind2, int ind3) = 0;
        int size_get(int s) const { return _new_size[s]; }
        const int* size_ptr_get() const { return _new_size; }
    protected:
      const int* _size;
      int _new_size[4];
      int _stot0;
      int _stot1;
    };

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
                const int* strides, int pad_top, int pad_left,
                FilterAccessor* faI, FilterAccessor* faK, int valid);

    void conv2d(const dbl_t* input, const dbl_t* kernel, dbl_t* out,
                const int* strides, int pad_top, int pad_left,
                const int* input_size, const int* kernel_size, int valid);

    /**
     * z is a 4D tensor output coming from a conv2d Op
     * bias is a vector with a value for each matrix in z
     * The result is the addition of each bias to its
     * corresponding matrix in z.
    */
    void conv2d_bias_add(const dbl_t* z, const dbl_t* bias, dbl_t* out,
                         const int* input_size);

    void conv2d_bias_add_grad(const dbl_t* z, const int* size, dbl_t* out);

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



    /**
     * Perform vector-vector substraction with coeff
     * out = coeff * (a - b)
     * a - vector (n)
     * b - vector (n)
     * out - vector (n)
     */
    void vect_sub_coeff(const dbl_t* a, const dbl_t* b, dbl_t coeff, dbl_t* out, 
                        std::size_t n);

    /**
     * Compute the gradient of the sigmoid operation
     * sig_out - vector (n)
     * dout - vector(n)
     * out - vector(n)
     *
     * let:
     * sig_out = sigmoid(z)
     * dout = nabla(E) / nabla(sig_out)
     * out = nabla(E) / nabla(Z)
     *sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
     *
     * out = sigmoid'(z) * dout
     *     = sig_out * (1 - sig_out) * dout
     */
    void sigmoid_grad(const dbl_t* sig_out, const dbl_t* dout, dbl_t* out,
                      std::size_t n);

    /**
     * perform matrix matrix multiplication followed by row vector addition
     * out = np.dot(x, w) + b
     * x - matrix (m * n)
     * w - matrix (n * p)
     * b - vector (p)
     * out - matrix (m * p)
     */
    void mat_mul_add(const dbl_t* x, const dbl_t* w, const dbl_t* b, dbl_t* out,
                     std::size_t m, std::size_t n, std::size_t p);


    /**
     * Perform the sum of each row of a matrix m
     * out = sum(m, axis=1)
     * m - matrix (m * n)
     * out - vector (m)
     */
    void mat_sum_rows(const dbl_t* a, dbl_t* out,
                      std::size_t m, std::size_t n);

    /**
     * Perform the sum of each col of a matrix m
     * out = sum(m, axis=0)
     * m - matrix (m * n)
     * out - vector (n)
     */
    void mat_sum_cols(const dbl_t* a, dbl_t* out,
                      std::size_t m, std::size_t n);


    /**
     * Cmpute the gradient of the cross-entrpy cost between y and softmax(logits) from logits
     * C = cross_entropy(y, softmax(logits))
     * y - matrix (m * n) - labels
     * logits - matrix (m * n) - logits of final layer (before y_hat = softmax(logits)
     * out - matrix (m * n)
     * out = nabla(C) / nabla(logits)
     */
    void softmax_cross_entropy_grad(const dbl_t* y, const dbl_t* logits, dbl_t* out,
                                    std::size_t m, std::size_t n);

    /**
     * Compute the gradient of the relu operation
     * z - vector (n)
     * dout - vector(n)
     * out - vector(n)
     *
     * let:
     * relu_out = relu(z)
     * dout = nabla(E) / nabla(relu_out)
     * out = nabla(E) / nabla(Z)
     * relu'(z) = 1 if z > 0, 0 otherwhise
     *
     * out = relu'(z) * dout
     *
     */
    void relu_grad(const dbl_t* z, const dbl_t* dout, dbl_t* out,
                   std::size_t n);

    /**
     * Perform vector-scalar addition on out
     * out = out + coeff * dv
     * out - vector(n)
     * dv - vector(n)
     * coeff - scalar
     */
    void vect_update(const dbl_t* dv, dbl_t* out, dbl_t coeff,
                     std::size_t n);

    /**
     * Compute sigmoid_cross_entropy
     * return sum(cross_entropy(y, sigmoid(logits))) / n
     * y - vector(n)
     * logits - vector(n)
     */
    dbl_t sigmoid_cross_entropy(const dbl_t* y, const dbl_t* logits,
                                std::size_t n);

    /**
     * Cmpute the gradient of the cross-entrpy cost between y and sigmoid(logits) from logits
     * C = cross_entropy(y, sigmoid(logits))
     * y - vector (n) - labels
     * logits - vector (n) - logits of final layer (before y_hat = sigmoid(logits)
     * out - vector (n)
     * out = nabla(C) / nabla(logits)
     */
    void sigmoid_cross_entropy_grad(const dbl_t* y, const dbl_t* logits, dbl_t* out,
                                    std::size_t n);

    /**
     * Compute the sum of equals(argmax(y), argmax(y_hat))
     * argmax(y) is done row by row
     * y - matrix(m * n)
     * y_hat - matrix (m * n)
     */
    std::size_t argmax_acc(const dbl_t* y, const dbl_t* y_hat,
                           std::size_t m, std::size_t n);

    class IdentityAccessor : public FilterAccessor
    {
    public:
        IdentityAccessor(const int* size)
                : FilterAccessor(size)
        {
          _stot0 = size[1] * size[2] * size[3];
          _stot1 = size[2] * size[3];
          _new_size[0] = size[0];
          _new_size[1] = size[1];
          _new_size[2] = size[2];
          _new_size[3] = size[3];
        }

        int access(int ind0, int ind1, int ind2, int ind3) override
        {
          int new_ind = ind0 * _stot0 + ind1 * _stot1
                        + ind2 * _size[3] + ind3;
          return new_ind;
        }
    };

    class WFilterRot180Accessor : public FilterAccessor
    {
    public:
        WFilterRot180Accessor(const int* size, int nbFilter, int nbChan)
              : FilterAccessor(size)
              , _nbFilter(nbFilter)
              , _nbChan(nbChan)
            {
              _new_size[0] = size[0];
              _new_size[1] = size[1];
              _new_size[2] = 1;
              _new_size[3] = 1;
            }
      int access(int ind0, int ind1, int, int) override
      {
          int new_ind = (_new_size[0] - ind0 - 1) * _stot0
                        + (_new_size[1] - ind1 - 1) * _stot1
                        + _nbChan * _size[3] + _nbFilter;
          return new_ind;
      }
    private:
      int _nbFilter;
      int _nbChan;
    };

    class YFilterAccessor : public FilterAccessor
    {
    public:
        YFilterAccessor(const int* size, int nbFilter)
              : FilterAccessor(size)
              , _nbFilter(nbFilter)
            {
              _new_size[0] = size[0];
              _new_size[1] = size[1];
              _new_size[2] = size[2];
              _new_size[3] = 1;
            }
      int access(int ind0, int ind1, int ind2, int) override
      {
          int new_ind = ind0 * _stot0 + ind1 * _stot1
                      + ind2 * _size[3] + _nbFilter;
          return new_ind;
      }
    private:
      int _nbFilter;
    };

    class ChFilterAccessor : public FilterAccessor
    {
    public:
        ChFilterAccessor(const int* size, int nbImg, int nbChan, int pad_top, int pad_left)
              : FilterAccessor(size)
              , _nbImg(nbImg)
              , _nbChan(nbChan)
              , _pad_top(pad_top)
              , _pad_left(pad_left)
            {
              _new_size[0] = 1;
              _new_size[1] = size[1];
              _new_size[2] = size[2];
              _new_size[3] = 1;
            }
      int access(int, int ind1, int ind2, int) override
      {
          if ((ind1 - _pad_top) >= 0
              && (ind2 - _pad_left) >= 0
              && (ind1 - _pad_top) < _size[1]
              && (ind2 - _pad_left) < _size[2])
          {
            int new_ind = _nbImg * _stot0 + ind1 * _stot1
                        + ind2 * _size[3] + _nbChan;
            return new_ind;
          }
          else
            return -1;
      }
    private:
      int _nbImg;
      int _nbChan;
      int _pad_top;
      int _pad_left;
    };

    class YtoKerAccessor : public FilterAccessor
    {
    public:
        YtoKerAccessor(const int* size, int outCh, int nbImage)
              : FilterAccessor(size)
              , _nbImage(nbImage)
              , _outCh(outCh)
            {
              _new_size[0] = size[1];
              _new_size[1] = size[2];
              _new_size[2] = size[3];
              _new_size[3] = outCh;
            }
      int access(int ind0, int ind1, int, int ind3) override
      {
          int new_ind = _nbImage * _stot0 + ind0 * _stot1 + ind1 * _size[3]
                        + ind3;
          return new_ind;
      }
    private:
      int _nbImage;
      int _outCh;
    };

    /**
     * Add padding around and in between an input
     * given a specific stride and kernel size.
     * WARNING: out must be allocated with calloc
     *          or filled with zeros.
     */
    void padd_full_conv(const dbl_t* input, dbl_t* out, int stride,
                        const int* input_size, const int* out_size,
                        FilterAccessor* faI);

    /**
     * Add padding in between a kernel
     * given a specific stride.
     * WARNING: out must be allocated with calloc
     *          or filled with zeros.
     */
    void padd_ker(const dbl_t* kernel, dbl_t* out, int stride,
                  const int* out_size, FilterAccessor* fa);

    /**
     * Format the calculated gradient for
     * the kernel to a standard format.
     * WARNING: dldw is freed at the end.
     */
    dbl_t* formatDw(dbl_t* dldw, const int* size);

    /**
     * Perform the operation t1 <- t1 + t2
     * on two 4D tensor.
     */
    void tensor_add(dbl_t* t1, dbl_t* t2, const int* size);

    /**
     * Concatenate two 4D tensor on the fourth axis
     * WARNING: t1 and t2 are freed.
     */
    dbl_t* tensor_concat_axis3(dbl_t* t1, dbl_t* t2, const int* size_t1, const int* size_t2);

    /**
     * Concatenate two 4D tensor on the first axis
     * WARNING: t1 and t2 are freed.
     */
    dbl_t* tensor_concat_axis0(dbl_t* t1, dbl_t* t2, const int* size_t1, const int* size_t2);

    struct padded_filter
    {
        dbl_t* data;
        int size[6];
        YFilterAccessor* acc;
    };
    void conv2d_input_grad(const dbl_t* dX1, const dbl_t* W1, const int stride, const int* dX1_size,
                           const int* W1_size, dbl_t* out, const int* input_size);

    struct padded_img
    {
        dbl_t* data;
        int img;
        int size[4];
        YtoKerAccessor* acc;
    };
    void conv2d_kernel_grad(const dbl_t* dX1, const dbl_t* X0, const int stride, const int* dX1_size,
                            const int* X0_size, dbl_t* out, const int* padded_size);


    /**
     * Perform moment update
     * out = a * out + b * dv
     * out - vector (n)
     * dv - vector (n)
     * a - scalar
     * b - scalar
     */
    void moment_update(const dbl_t* dv, dbl_t* out,
                       dbl_t a, dbl_t b, std::size_t len);

    /**
     * Perform squared moment update
     * out = a * out + b * dv * dv
     * out - vector (n)
     * dv - vector (n)
     * a - scalar
     * b - scalar
     */
    void moment_update2(const dbl_t* dv, dbl_t* out,
                        dbl_t a, dbl_t b, std::size_t len);


    /**
     * Compute the gradient of the leaky_relu operation
     * z - vector (n)
     * dout - vector(n)
     * out - vector(n)
     *
     * let:
     * relu_out = relu(z)
     * dout = nabla(E) / nabla(leaku_relu_out)
     * out = nabla(E) / nabla(Z)
     * leaky_relu'(z) = 1 if z > 0, alpha otherwhise
     *
     * out = leaky_relu'(z) * dout
     *
     */
    void leaky_relu_grad(const dbl_t* z, const dbl_t* dout, dbl_t* out,
                         dbl_t alpha, std::size_t n);
  
    /**
       * Compute the gradient of the tanh operation
       * tanh_out - vector (n)
       * dout - vector(n)
       * out - vector(n)
       *
       * let:
       * tanh_out = tanh(z)
       * dout = nabla(E) / nabla(tanh_out)
       * out = nabla(E) / nabla(Z)
       * tanh'(z) = 1 - tanh(z)^2
       *
       * out = tanh'(z) * dout
       *     = (1 - tanh(z)^2) * dout
       */
      void tanh_grad(const dbl_t* tanh_out, const dbl_t* dout, dbl_t* out,
                        std::size_t n);
}

#include "ops.hxx"