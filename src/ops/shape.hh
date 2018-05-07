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
        bool defined() const;
        int total() const;

    private:
        std::vector<int> dims_;

    };

    bool operator==(const Shape& a, const Shape& b);
    bool operator!=(const Shape& a, const Shape& b);

    std::ostream& operator<<(std::ostream& os, const Shape& s);
    
    
}
