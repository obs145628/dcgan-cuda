#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct Tensor4
{

    static std::vector<Tensor4> load_tensors(const std::string& path);
    static void save_tensors(const std::string& path, const std::vector<Tensor4>& tensors);

    Tensor4(std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4);
    Tensor4(const Tensor4& t);
    Tensor4(Tensor4&& t);
    ~Tensor4();
    Tensor4& operator=(const Tensor4&) = delete;
    Tensor4& operator=(Tensor4&&) = delete;

    float& operator()(std::size_t i1, std::size_t i2,
                      std::size_t i3, std::size_t i4);
    const float& operator()(std::size_t i1, std::size_t i2,
                      std::size_t i3, std::size_t i4) const;

    void dump_shape() const;

    /**
     * Create a new tensor, padded horizontally and vertically 
     * of ph and pw dims of 0, on both sides
     */
    Tensor4 pad0(std::size_t ph, std::size_t pw) const;

    const std::size_t d1;
    const std::size_t d2;
    const std::size_t d3;
    const std::size_t d4;
    const std::size_t size;
    float* data;   
};
