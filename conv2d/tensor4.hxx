#include "tensor4.hh"
#include <cassert>

inline Tensor4::Tensor4(Tensor4&& t)
    : d1(t.d1)
    , d2(t.d2)
    , d3(t.d3)
    , d4(t.d4)
    , size(t.size)
    , data(t.data)
{
    t.data = nullptr;
}

inline float& Tensor4::operator()(std::size_t i1, std::size_t i2,
                                  std::size_t i3, std::size_t i4)
{
    assert(i1 < d1);
    assert(i2 < d2);
    assert(i3 < d3);
    assert(i4 < d4);
    return data[
        i1 * d2 * d3 * d4 + i2 * d3 * d4 + i3 * d4 + i4
        ];
}

inline const float& Tensor4::operator()(std::size_t i1, std::size_t i2,
                                        std::size_t i3, std::size_t i4) const
{
    assert(i1 < d1);
    assert(i2 < d2);
    assert(i3 < d3);
    assert(i4 < d4);
    return data[
        i1 * d2 * d3 * d4 + i2 * d3 * d4 + i3 * d4 + i4
        ];
}
