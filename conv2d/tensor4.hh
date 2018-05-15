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
     * The tensor must be an input (width and height in dim2 and dimd3)
     */
    Tensor4 pad0(std::size_t ph, std::size_t pw) const;

    /**
     * Create a new tensor, padded of p1, p2, p3 p4 
     * respectictly top, bottom, left, right
     * The tensor must be an input (width and height in dim2 and dimd3)
     */
    Tensor4 pad0(std::size_t p1, std::size_t p2, std::size_t p3, std::size_t p4) const;

    /**
     * Create a new tensor, with a new shape
     * the new size must be equal to the old one
     */
    Tensor4 reshape(std::size_t nd1, std::size_t nd2, std::size_t nd3, std::size_t nd4) const;

    /**
     * Create a new tensor, with its dimensions transposed
     */
    Tensor4 transpose(std::size_t t1, std::size_t t2, std::size_t t3, std::size_t t4) const;


    /**
     * Create a new tensor, with it's width and height rotated by 180
     * The actual tensor is a filter (height and width are in dim 1 and 2)
     */
    Tensor4 frot180() const;

    /**
     * Create a new tensor, with each lines seperated by h lines of 0,
     * and it's columns seperated h w columns of 0
     * The actual tensor is a filter (height and width are in dim 1 and 2)
     */
    Tensor4 fstride0(std::size_t h, std::size_t w) const;

    /**
     * Create a new tensor, with each lines seperated by h lines of 0,
     * and it's columns seperated h w columns of 0
     * The actual tensor is an input (height and width are d2 and d3)
     */
    Tensor4 istride0(std::size_t h, std::size_t w) const;

    /**
     * Extract only one 2d region
     * The actual tensor is an input (height and width are d2 and d3)
     */
    Tensor4 iregion(std::size_t y, std::size_t x, std::size_t h, std::size_t w) const;

    /**
     * Dump a filter in 2 dimensions (height and width are d1 and 2)
     */
    void fdump_2d() const;

    const std::size_t d1;
    const std::size_t d2;
    const std::size_t d3;
    const std::size_t d4;
    const std::size_t size;
    float* data;   
};

#include "tensor4.hxx"
