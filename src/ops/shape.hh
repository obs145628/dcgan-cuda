#pragma once

#include <iostream>
#include <vector>

namespace ops
{

    class Shape
    {
    public:
        Shape(const std::vector<int>& dims = {});

        const std::vector<int>& dims() const;
        std::size_t ndims() const;
        int operator[](std::size_t i) const;

        /**
         * Returns true if the shape is whole (no -1)
         */
        bool defined() const;

        /**
         * product of all sizes
         * total size of the tensor
         */
        int total() const;

        /**
         * Compute the transpose shape
         * Works only on matrices
         */
        Shape transpose() const;

    private:
        std::vector<int> dims_;

    };

    bool operator==(const Shape& a, const Shape& b);
    bool operator!=(const Shape& a, const Shape& b);

    std::ostream& operator<<(std::ostream& os, const Shape& s);
    
    
}
