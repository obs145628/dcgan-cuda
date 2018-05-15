#include "tensor4.hh"
#include <algorithm>
#include <array>
#include <cassert>
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
    assert(i1 < d1);
    assert(i2 < d2);
    assert(i3 < d3);
    assert(i4 < d4);
    return data[
        i1 * d2 * d3 * d4 + i2 * d3 * d4 + i3 * d4 + i4
        ];
}

const float& Tensor4::operator()(std::size_t i1, std::size_t i2,
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

Tensor4 Tensor4::reshape(std::size_t nd1, std::size_t nd2, std::size_t nd3, std::size_t nd4) const
{
    Tensor4 res(nd1, nd2, nd3, nd4);
    assert(res.size == size);
    std::copy(data, data + size, res.data);
    return res;
}

Tensor4 Tensor4::transpose(std::size_t t1, std::size_t t2, std::size_t t3, std::size_t t4) const
{
    std::vector<std::size_t> odims{d1, d2, d3, d4};
    Tensor4 res(odims[t1], odims[t2], odims[t3], odims[t4]);

    std::vector<std::size_t> idx (4);
    idx[t1] = 0;
    idx[t2] = 1;
    idx[t3] = 2;
    idx[t4] = 3;
    t1 = idx[0];
    t2 = idx[1];
    t3 = idx[2];
    t4 = idx[3];

    for (std::size_t i1 = 0; i1 < res.d1; ++i1)
        for (std::size_t i2 = 0; i2 < res.d2; ++i2)
            for (std::size_t i3 = 0; i3 < res.d3; ++i3)
                for (std::size_t i4 = 0; i4 < res.d4; ++i4)
                {
                    std::array<std::size_t, 4> pos {i1, i2, i3, i4};
                    const auto& x = (*this)(pos[t1], pos[t2], pos[t3], pos[t4]);
                    res(i1, i2, i3, i4) = x;
                }

    return res;
}


Tensor4 Tensor4::fstride0(std::size_t h, std::size_t w) const
{
    Tensor4 res(1 + (h + 1) * (d1 - 1), 1 + (w + 1) * (d2 - 1), d3, d4);
    std::fill(res.data, res.data + res.size, 0);

    for (std::size_t i1 = 0; i1 < d1; ++i1)
        for (std::size_t i2 = 0; i2 < d2; ++i2)
            for (std::size_t i3 = 0; i3 < d3; ++i3)
                for (std::size_t i4 = 0; i4 < d4; ++i4)
                    res(i1 * (h + 1), i2 * (w + 1), i3, i4) = (*this)(i1, i2, i3, i4);

    return res;
}

Tensor4 Tensor4::istride0(std::size_t h, std::size_t w) const
{
    Tensor4 res(d1, 1 + (h + 1) * (d2 - 1), 1 + (w + 1) * (d3 - 1), d4);
    std::fill(res.data, res.data + res.size, 0);

    for (std::size_t i1 = 0; i1 < d1; ++i1)
        for (std::size_t i2 = 0; i2 < d2; ++i2)
            for (std::size_t i3 = 0; i3 < d3; ++i3)
                for (std::size_t i4 = 0; i4 < d4; ++i4)
                    res(i1, i2 * (h + 1), i3 * (w + 1), i4) = (*this)(i1, i2, i3, i4);

    return res;
}

void Tensor4::fdump_2d() const
{
    for (std::size_t i1 = 0; i1 < d1; ++i1)
    {
        for (std::size_t i2 = 0; i2 < d2; ++i2)
        {
            std::cout << (*this)(i1, i2, 0, 0) << "|";
        }
        std::cout << "\n";
    }
}
