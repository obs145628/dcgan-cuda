#pragma once

#include <cstddef>
#include "tensor4.hh"

namespace acc
{


    template <class T>
    struct Tensor4
    {
    public:

        using elem_t = T;

        /**
         * Type traits
         *
         */
        static constexpr bool IsDense = true; //all it's elements are defined
            
        Tensor4(T ptr,
                std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4)
            : ptr_(ptr)
            , d1_(d1)
            , d2_(d2)
            , d3_(d3)
            , d4_(d4)
            , size_(d1*d2*d3*d4)
            {}

    private:
        T ptr_;
        const std::size_t d1_;
        const std::size_t d2_;
        const std::size_t d3_;
        const std::size_t d4_;
        const std::size_t size_;

    public:


        T ptr() const
            {
                return ptr_;
            }


        std::size_t d1() const
            {
                return d1_;
            }

        std::size_t d2() const
            {
                return d2_;
            }


        std::size_t d3() const
            {
                return d3_;
            }


        std::size_t d4() const
            {
                return d4_;
            }

        std::size_t size() const
            {
                return size_;
            }
            
        T operator()(std::size_t i1, std::size_t i2,
                     std::size_t i3, std::size_t i4) const
            {

                std::size_t idx = i1 * d2_ * d3_ * d4_ + i2 * d3_ * d4_ + i3 * d4_ + i4;
                return ptr_ + idx;
            }
    };

    template <class T>
    struct Tensor4Pad
    {
    public:
        Tensor4Pad(T ptr,
                   std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4,
                   std::size_t ptop, std::size_t pleft, std::size_t pbot,
                   std::size_t pright)
            : ptr_(ptr)
            , d1_(d1)
            , d2_(d2)
            , d3_(d3)
            , d4_(d4)
            , size_(d1*d2*d3*d4)
            , ptop_(ptop)
            , pleft_(pleft)
            , pbot_(pbot)
            , pright_(pright)
            {}

    public:
        T ptr_;
        const std::size_t d1_;
        const std::size_t d2_;
        const std::size_t d3_;
        const std::size_t d4_;
        const std::size_t size_;
        const std::size_t ptop_;
        const std::size_t pleft_;
        const std::size_t pbot_;
        const std::size_t pright_;

    public:

        using elem_t = T;
            
        /**
         * Type traits
         *
         */
        static constexpr bool IsDense = false; //0 on border

        
        T ptr()
            {
                return ptr_;
            }

        
        std::size_t d1() const
            {
                return d1_;
            }

        
        std::size_t d2() const
            {
                return d2_ + ptop_ + pbot_;
            }

        
        std::size_t d3() const
            {
                return d3_ + pleft_ + pright_;
            }

        
        std::size_t d4() const
            {
                return d4_;
            }

        
        std::size_t size() const
            {
                return d1() * d2() * d3() * d4();
            }
            
        
        T operator()(std::size_t i1, std::size_t i2,
                     std::size_t i3, std::size_t i4) const
            {
                i2 -= ptop_;
                i3 -= pleft_;
                if (i2 >= d2_ || i3 >= d3_)
                    return nullptr;
                
                std::size_t idx = i1 * d2_ * d3_ * d4_ + i2 * d3_ * d4_ + i3 * d4_ + i4;
                return ptr_ + idx;
            }
    };


    
    template <class T>
    struct Tensor4Tr3124
    {
    public:

        using elem_t = T;

        /**
         * Type traits
         *
         */
        static constexpr bool IsDense = true; //all it's elements are defined
            
        Tensor4Tr3124(T ptr,
                      std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4)
            : ptr_(ptr)
            , d1_(d1)
            , d2_(d2)
            , d3_(d3)
            , d4_(d4)
            , size_(d1*d2*d3*d4)
            {}

    private:
        T ptr_;
        const std::size_t d1_;
        const std::size_t d2_;
        const std::size_t d3_;
        const std::size_t d4_;
        const std::size_t size_;

    public:


        T ptr() const
            {
                return ptr_;
            }


        std::size_t d1() const
            {
                return d3_;
            }

        std::size_t d2() const
            {
                return d1_;
            }


        std::size_t d3() const
            {
                return d2_;
            }


        std::size_t d4() const
            {
                return d4_;
            }

        std::size_t size() const
            {
                return size_;
            }
            
        T operator()(std::size_t i1, std::size_t i2,
                     std::size_t i3, std::size_t i4) const
            {
                std::size_t idx = i2 * d2_ * d3_ * d4_ + i3 * d3_ * d4_ + i1 * d4_ + i4;
                return ptr_ + idx;
            }
    };


    template <class T>
    struct Tensor4DkX
    {
    public:

        Tensor4DkX(T ptr,
                   std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4,
                   std::size_t ptop, std::size_t pleft, std::size_t pbot,
                   std::size_t pright)
            : ptr_(ptr)
            , d1_(d1)
            , d2_(d2)
            , d3_(d3)
            , d4_(d4)
            , size_(d1*d2*d3*d4)
            , ptop_(ptop)
            , pleft_(pleft)
            , pbot_(pbot)
            , pright_(pright)
            {}

    public:
        T ptr_;
        const std::size_t d1_;
        const std::size_t d2_;
        const std::size_t d3_;
        const std::size_t d4_;
        const std::size_t size_;
        const std::size_t ptop_;
        const std::size_t pleft_;
        const std::size_t pbot_;
        const std::size_t pright_;

    public:

        using elem_t = T;
            
        /**
         * Type traits
         *
         */
        static constexpr bool IsDense = false; //0 on border

    public:


        T ptr() const
            {
                return ptr_;
            }


        std::size_t d1() const
            {
                return d4_;
            }

        std::size_t d2() const
            {
                return d2_ + ptop_ + pbot_;
            }

        
        std::size_t d3() const
            {
                return d3_ + pleft_ + pright_;
            }


        std::size_t d4() const
            {
                return d1_;
            }

        std::size_t size() const
            {
                return d1() * d2() * d3() * d4();
            }
            
        T operator()(std::size_t i1, std::size_t i2,
                     std::size_t i3, std::size_t i4) const
            {
                i2 -= ptop_;
                i3 -= pleft_;
                if (i2 >= d2_ || i3 >= d3_)
                    return nullptr;
                
                std::size_t idx = i4 * d2_ * d3_ * d4_ + i2 * d3_ * d4_ + i3 * d4_ + i1;
                return ptr_ + idx;
            }
    };

    template <class T>
    struct Tensor4DkDy
    {
    public:

        using elem_t = T;

        /**
         * Type traits
         *
         */
        static constexpr bool IsDense = false; //strided with 0
            
        Tensor4DkDy(T ptr,
                    std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4,
                    std::size_t h, std::size_t w)
            : ptr_(ptr)
            , d1_(d1)
            , d2_(d2)
            , d3_(d3)
            , d4_(d4)
            , h_(h)
            , w_(w)
            , size_(d1*d2*d3*d4)
            {}

    private:
        T ptr_;
        const std::size_t d1_;
        const std::size_t d2_;
        const std::size_t d3_;
        const std::size_t d4_;
        const std::size_t h_;
        const std::size_t w_;
        const std::size_t size_;

    public:


        T ptr() const
            {
                return ptr_;
            }


        std::size_t d1() const
            {
                return 1 + (h_ + 1) * (d2_ - 1);
            }

        std::size_t d2() const
            {
                return 1 + (w_ + 1) * (d3_ - 1);
            }


        std::size_t d3() const
            {
                return d1_;
            }


        std::size_t d4() const
            {
                return d4_;
            }

        std::size_t size() const
            {
                return d1() * d2() * d3() * d4();
            }
            
        T operator()(std::size_t i1, std::size_t i2,
                     std::size_t i3, std::size_t i4) const
            {

                if (i1 % (h_ + 1) || i2 % (w_ + 1))
                    return nullptr;

                i1 /= (h_ + 1);
                i2 /= (w_ + 1);

                std::size_t idx = i3 * d2_ * d3_ * d4_ + i1 * d3_ * d4_ + i2 * d4_ + i4;
                return ptr_ + idx;
            }
    };


    template <class T>
    float t_get(const T& tensor,
                std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4)
    {
        auto ptr = tensor(i1, i2, i3, i4);
        if (T::IsDense || ptr)
            return *ptr;
        else
            return 0;
    }

    template <class T>
    void t_set(const T& tensor,
               std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4,
               float val)
    {
        auto ptr = tensor(i1, i2, i3, i4);
        if (T::IsDense || ptr)
            *ptr = val;
    }

    /**
     * Compute a convolution with padding and strides
     * input (i1, h1, w1, c1) nb input * input height * input width * input channels
     * filter (fh, fw, c1, k) filter height * filter with * input channels * nb filters
     * sh - height of strides
     * sw - width of strides
     * p1 - top padding
     * p2 - bottom padding
     * p3 - left padding
     * p4 - right padding
     * out (i1, (h1 - fh + p1 + p2) / sh + 1, (w1 - fw + p3 + p4) / sw + 1, k)
     */
    ::Tensor4 conv2d_sp(const ::Tensor4& input, const ::Tensor4& filter,
                      std::size_t sh, std::size_t sw,
                      std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);

    void conv2d_sp(const float* x, const float* k, float* y,
                   std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                   std::size_t hk, std::size_t wk, std::size_t ck,
                   std::size_t sh, std::size_t sw,
                   std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);

    /**
     * Compute a convolution with padding and strides
     * input (i1, h1, w1, c1) nb input * input height * input width * input channels
     * filter (fh, fw, c1, k) filter height * filter with * input channels * nb filters
     * dout (i1, (h1 - fh + p1 + p2) / sh + 1, (w1 - fw + p3 + p4) / sw + 1, k)
     * @return dfilter (fh, fw, c1, k)
     * sh - height of strides
     * sw - width of strides
     * p1 - top padding
     * p2 - bottom padding
     * p3 - left padding
     * p4 - right padding
     */
    ::Tensor4 conv2d_sp_dk(const ::Tensor4& input, const ::Tensor4& dout,
                           std::size_t sh, std::size_t sw,
                           std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);

    void conv2d_sp_dk(const float* x, const float* dy, float* dk,
                      std::size_t nx, std::size_t hx, std::size_t wx, std::size_t cx,
                      std::size_t hy, std::size_t wy, std::size_t cy,
                      std::size_t sh, std::size_t sw,
                      std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4);
}
