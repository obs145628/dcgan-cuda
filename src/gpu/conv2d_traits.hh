#pragma once

namespace gpu
{

    namespace
    {
        template <std::size_t D1,
                  std::size_t D2,
                  std::size_t D3,
                  std::size_t D4>
        struct Tensor4Trait
        {
            static constexpr std::size_t d1 = D1;
            static constexpr std::size_t d2 = D2;
            static constexpr std::size_t d3 = D3;
            static constexpr std::size_t d4 = D4;
        };

        template <std::size_t D1,
                  std::size_t D2,
                  std::size_t D3,
                  std::size_t D4,
                  std::size_t PadTop,
                  std::size_t PadLeft,
                  std::size_t PadBot,
                  std::size_t PadRight>
        struct Tensor4IPadTrait
        {
            static constexpr std::size_t d1 = D1;
            static constexpr std::size_t d2 = D2 + PadTop + PadBot;
            static constexpr std::size_t d3 = D3 + PadLeft + PadRight;
            static constexpr std::size_t d4 = D4;
            static constexpr std::size_t pad_top = PadTop;
            static constexpr std::size_t pad_left = PadLeft;
            static constexpr std::size_t pad_bot = PadBot;
            static constexpr std::size_t pad_right = PadRight;
        };

        template <class X, class K, class Y,
                  std::size_t SH,
                  std::size_t SW>
        struct ConvTrait
        {
            using TX = X;
            using TK = K;
            using TY = Y;
            static constexpr std::size_t sh = SH;
            static constexpr std::size_t sw = SW;


            static constexpr std::size_t nb_images = X::d1;
            static constexpr std::size_t x_width = X::d2;
            static constexpr std::size_t x_height = X::d3;
            static constexpr std::size_t in_chans = X::d4;
            static constexpr std::size_t out_chans = K::d4;
            static constexpr std::size_t k_width = K::d1;
            static constexpr std::size_t k_height = K::d2;
            static constexpr std::size_t y_width = Y::d2;
            static constexpr std::size_t y_height = Y::d3;
        };
        
    }
    
}
