#pragma once

namespace gpu
{

    namespace
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
            
            __host__
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

            __device__ __host__
            T ptr() const
                {
                    return ptr_;
                }

            __device__ __host__
            std::size_t d1() const
                {
                    return d1_;
                }

            __device__ __host__
            std::size_t d2() const
                {
                    return d2_;
                }

            __device__ __host__
            std::size_t d3() const
                {
                    return d3_;
                }

            __device__ __host__
            std::size_t d4() const
                {
                    return d4_;
                }

            __device__ __host__
            std::size_t size() const
                {
                    return size_;
                }
            
            __device__ T operator()(std::size_t i1, std::size_t i2,
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
            __device__ __host__
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

            __device__ __host__
            T ptr()
                {
                    return ptr_;
                }

            __device__ __host__
            std::size_t d1() const
                {
                    return d1_;
                }

            __device__ __host__
            std::size_t d2() const
                {
                    return d2_ + ptop_ + pbot_;
                }

            __device__ __host__
            std::size_t d3() const
                {
                    return d3_ + pleft_ + pright_;
                }

            __device__ __host__
            std::size_t d4() const
                {
                    return d4_;
                }

            __device__ __host__
            std::size_t size() const
                {
                    return d1() * d2() * d3() * d4();
                }
            
            __device__
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

            __host__
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

            __device__ __host__
            T ptr() const
                {
                    return ptr_;
                }

            __device__ __host__
            std::size_t d1() const
                {
                    return d3_;
                }
            
            __device__ __host__
            std::size_t d2() const
                {
                    return d1_;
                }

            __device__ __host__
            std::size_t d3() const
                {
                    return d2_;
                }

            __device__ __host__
            std::size_t d4() const
                {
                    return d4_;
                }

            std::size_t size() const
                {
                    return size_;
                }

            __device__
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
            __host__
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

            __device__ __host__
            T ptr() const
                {
                    return ptr_;
                }

            __device__ __host__
            std::size_t d1() const
                {
                    return d4_;
                }

            __device__ __host__
            std::size_t d2() const
                {
                    return d2_ + ptop_ + pbot_;
                }

            __device__ __host__
            std::size_t d3() const
                {
                    return d3_ + pleft_ + pright_;
                }

            __device__ __host__
            std::size_t d4() const
                {
                    return d1_;
                }

            __device__ __host__
            std::size_t size() const
                {
                    return d1() * d2() * d3() * d4();
                }

            __device__
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

            __host__
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

            __device__ __host__
            T ptr() const
                {
                    return ptr_;
                }

            __device__ __host__
            std::size_t d1() const
                {
                    return 1 + (h_ + 1) * (d2_ - 1);
                }

            __device__ __host__
            std::size_t d2() const
                {
                    return 1 + (w_ + 1) * (d3_ - 1);
                }

            __device__ __host__
            std::size_t d3() const
                {
                    return d1_;
                }

            __device__ __host__
            std::size_t d4() const
                {
                    return d4_;
                }

            __device__ __host__
            std::size_t size() const
                {
                    return d1() * d2() * d3() * d4();
                }

            __device__
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
        struct Tensor4DxDk
        {
        public:

            using elem_t = T;

            /**
             * Type traits
             *
             */
            static constexpr bool IsDense = true; //all it's elements are defined

            __host__
            Tensor4DxDk(T ptr,
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

            __device__ __host__
            T ptr() const
                {
                    return ptr_;
                }

            __device__ __host__
            std::size_t d1() const
                {
                    return d1_;
                }

            __device__ __host__
            std::size_t d2() const
                {
                    return d2_;
                }

            __device__ __host__
            std::size_t d3() const
                {
                    return d4_;
                }

            __device__ __host__
            std::size_t d4() const
                {
                    return d3_;
                }

            __device__ __host__
            std::size_t size() const
                {
                    return size_;
                }

            __device__
            T operator()(std::size_t i1, std::size_t i2,
                         std::size_t i3, std::size_t i4) const
                {

                    i1 = d1_ - i1 - 1;
                    i2 = d2_ - i2 - 1;
                    std::size_t idx = i1 * d2_ * d3_ * d4_ + i2 * d3_ * d4_ + i4 * d4_ + i3;
                    return ptr_ + idx;
                }
        };


        template <class T>
        struct Tensor4DxDy
        {
        public:

            __host__
            Tensor4DxDy(T ptr,
                        std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4,
                        std::size_t ptop, std::size_t pleft, std::size_t pbot,
                        std::size_t pright,
                        std::size_t h,
                        std::size_t w)
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
                , h_(h)
                , w_(w)
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
            const std::size_t h_;
            const std::size_t w_;

        public:

            using elem_t = T;
            
            /**
             * Type traits
             *
             */
            static constexpr bool IsDense = false; //0 on border and with strides

            __device__ __host__
            T ptr()
                {
                    return ptr_;
                }

            __device__ __host__
            std::size_t d1() const
                {
                    return d1_;
                }
        
            __device__ __host__
            std::size_t d2() const
                {
                    return 1 + (h_ + 1) * (d2_ - 1) + ptop_ + pbot_;
                }

            __device__ __host__
            std::size_t d3() const
                {
                    return 1 + (w_ + 1) * (d3_ - 1) + pleft_ + pright_;
                }

            __device__ __host__
            std::size_t d4() const
                {
                    return d4_;
                }

            __device__ __host__
            std::size_t size() const
                {
                    return d1() * d2() * d3() * d4();
                }
            
            __device__
            T operator()(std::size_t i1, std::size_t i2,
                         std::size_t i3, std::size_t i4) const
                {
                    i2 -= ptop_;
                    i3 -= pleft_;

                    std::size_t stride_d2 = 1 + (h_ + 1) * (d2_ - 1);
                    std::size_t stride_d3 = 1 + (w_ + 1) * (d3_ - 1);
                
                    if (i2 >= stride_d2 || i3 >= stride_d3)
                        return nullptr;
                
                    if (i2 % (h_ + 1) || i3 % (w_ + 1))
                        return nullptr;

                    i2 /= (h_ + 1);
                    i3 /= (w_ + 1);
                
                    std::size_t idx = i1 * d2_ * d3_ * d4_ + i2 * d3_ * d4_ + i3 * d4_ + i4;
                    return ptr_ + idx;
                }
        };


        template <class T>
        __device__ dbl_t t_get(const T& tensor,
                               std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4)
        {
            auto ptr = tensor(i1, i2, i3, i4);
            if (T::IsDense || ptr)
                return *ptr;
            else
                return 0;
        }

        template <class T>
        __device__ void t_set(const T& tensor,
                              std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4,
                              dbl_t val)
        {
            auto ptr = tensor(i1, i2, i3, i4);
            if (T::IsDense || ptr)
                *ptr = val;
        }
        
    }
    
}
