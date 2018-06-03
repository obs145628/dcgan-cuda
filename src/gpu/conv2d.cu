#include "../runtime/node.hh"
#include <iostream>


namespace gpu
{

    namespace
    {

        constexpr std::size_t BLOCK_SIZE = 512;

        template <class T>
        struct Tensor4
        {
        public:

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

            /**
             * Type traits
             *
             */
            static constexpr bool IsDense = false; //0 on border

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
        

        template <class T1, class T2>
        __device__ dbl_t compute_val(const T1& x, const T2& k,
                                     std::size_t d1, std::size_t d2,
                                     std::size_t d3, std::size_t d4,
                                     std::size_t sh, std::size_t sw)
        {
            dbl_t res = 0;
            for (std::size_t i1 = 0; i1 < k.d1(); ++i1)
                for (std::size_t i2 = 0; i2 < k.d2(); ++i2)
                    for (std::size_t i3 = 0; i3 < k.d3(); ++i3)
                    {
                        
                        dbl_t vx = t_get(x, d1, d2*sh + i1, d3*sw + i2, i3);
                        dbl_t vk = t_get(k, i1, i2, i3, d4);
                        res += vx * vk;
                    }
            return res;
        }

        template <class T1, class T2, class T3>
        __global__ void conv2d(const T1 x, const T2 k, const T3 y,
                               std::size_t sh, std::size_t sw)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= y.size())
                return;
            
            std::size_t y_i1 = index / (y.d2() * y.d3() * y.d4());
            std::size_t y_i2 = (index % (y.d2() * y.d3() * y.d4())) / (y.d3() * y.d4());
            std::size_t y_i3 = (index % (y.d3() * y.d4())) / y.d4();
            std::size_t y_i4 = index % y.d4();
            
            t_set(y, y_i1, y_i2, y_i3, y_i4,
                  compute_val(x, k,
                              y_i1, y_i2, y_i3, y_i4, sh, sw));
        }
        
    }

    void kernel_conv2d(rt::Node* node)
    {

        const dbl_t* x = node->in1;
        std::size_t nx = node->sizes1[0];
        std::size_t hx = node->sizes1[1];
        std::size_t wx = node->sizes1[2];
        std::size_t cx = node->sizes1[3];
        
        const dbl_t* k = node->in2;
        std::size_t hk = node->sizes2[0];
        std::size_t wk = node->sizes2[1];
        std::size_t ck = node->sizes2[3];
        
        dbl_t* y = node->out1;
        std::size_t sh = node->intconst[0];
        std::size_t sw = node->intconst[1];
        int pad_height = node->int_cons1;
        int pad_width = node->int_cons2;

        std::size_t pad_top = pad_height / 2;
        std::size_t pad_left = pad_width / 2;
        std::size_t pad_bot = pad_height - pad_top;
        std::size_t pad_right = pad_width - pad_left;

        std::size_t hy = (hx + pad_top + pad_bot - hk) / sh + 1;
        std::size_t wy = (wx + pad_left + pad_right - wk) / sw + 1;

        std::cout << "X: " << nx << ", " << hx << ", " << wx << ", " << cx << std::endl;
        std::cout << "K: " << hk << ", " << wk << ", " << cx << ", " << ck << std::endl;
        std::cout << "S: " << sh << ", " << sw << std::endl;
        
        std::cout << "P: " << pad_top << ", " << pad_bot << ", " << pad_left << ", " << pad_right
                  << std::endl;

        std::cout << "Y: " << nx << ", " << hy << ", " << wy << ", " << ck << std::endl;

        

        /*
        conv2d_sp(input, input_size[0], input_size[1], input_size[2], input_size[3],
                  kernel, kernel_size[0], kernel_size[1], kernel_size[3],
                  out, strides[0], strides[1], pad_top, pad_bot, pad_left, pad_right);
        */

        Tensor4Pad<const dbl_t*> tx(x, nx, hx, wx, cx,
                                    pad_top, pad_left, pad_bot, pad_right);
        Tensor4<const dbl_t*> tk(k, hk, wk, cx, ck);
        Tensor4<dbl_t*> ty(y, nx, hy, wy, ck);

        std::size_t len = ty.size();
        std::size_t nb_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

        std::cout << "nb blocks = " << nb_blocks << std::endl;

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        conv2d<<<nb_blocks, BLOCK_SIZE>>>(tx, tk, ty, sh, sw);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);

        std::cout << "time = " << time << "ms\n" << std::endl;
    }
    
}
