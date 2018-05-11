#include "tensor4.hh"
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <tocha/tensor.hh>

std::vector<Tensor4> Tensor4::load_tensors(const std::string& path)
{
    auto data = tocha::Tensors::load(path);
    std::vector<Tensor4> res;

    for (auto& tmp: data.arr())
    {
        if (tmp.dims.size() != 4)
            throw std::runtime_error {"not a tensor of size 4"};

        Tensor4 t(tmp.dims[0], tmp.dims[1], tmp.dims[2], tmp.dims[3]);
        auto ptr = reinterpret_cast<float*>(tmp.data);
        std::copy(ptr, ptr + t.size, t.data);
        res.push_back(t);
    }

    return res;
}

void Tensor4::save_tensors(const std::string& path, const std::vector<Tensor4>& tensors)
{
    tocha::Tensors out;

    for (auto& t : tensors)
    {
        out.add(tocha::Tensor::f32(t.d1, t.d2, t.d3, t.d4));
        auto ptr = reinterpret_cast<float*>(out.arr().back().data);
        std::copy(t.data, t.data + t.size, ptr);
    }

    out.save(path);
}

Tensor4::Tensor4(std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4)
    : d1(d1)
    , d2(d2)
    , d3(d3)
    , d4(d4)
    , size(d1 * d2 * d3 * d4)
    , data(new float[size])
{}

Tensor4::Tensor4(const Tensor4& t)
    : Tensor4(t.d1, t.d2, t.d3, t.d4)
{
    std::copy(t.data, t.data + t.size, data);
}

Tensor4::Tensor4(Tensor4&& t)
    : d1(t.d1)
    , d2(t.d2)
    , d3(t.d3)
    , d4(t.d4)
    , size(t.size)
    , data(t.data)
{
    t.data = nullptr;
}

float& Tensor4::operator()(std::size_t i1, std::size_t i2,
                           std::size_t i3, std::size_t i4)
{
    return data[
        i1 * d2 * d3 * d4 + i2 * d3 * d4 + i3 * d4 + i4
        ];
}

const float& Tensor4::operator()(std::size_t i1, std::size_t i2,
                                 std::size_t i3, std::size_t i4) const
{
    return data[
        i1 * d2 * d3 * d4 + i2 * d3 * d4 + i3 * d4 + i4
        ];
}

Tensor4::~Tensor4()
{
    delete[] data;
}


void Tensor4::dump_shape() const
{
    std::cout << "(" << d1 << ", " << d2 << ", "
              << d3 << ", " << d4 << ")" << std::endl;
}

Tensor4 Tensor4::pad0(std::size_t ph, std::size_t pw) const
{
    Tensor4 res(d1, d2 + 2 * ph, d3 + 2 * pw, d4);
    std::fill(res.data, res.data + res.size, 0);

    for (std::size_t i1 = 0; i1 < d1; ++i1) 
        for (std::size_t i2 = 0; i2 < d2; ++i2)
            for (std::size_t i3 = 0; i3 < d3; ++i3)
                for (std::size_t i4 = 0; i4 < d4; ++i4)
                    res(i1, i2 + ph, i3 + pw, i4) = (*this)(i1, i2, i3, i4);

    return res;
}
