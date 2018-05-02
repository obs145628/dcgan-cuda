#include "shape.hh"

namespace ops
{

    Shape::Shape(const std::vector<int>& dims)
	: dims_(dims)
    {}

    const std::vector<int>& Shape::dims() const
    {
	return dims_;
    }

    std::size_t Shape::ndims() const
    {
	return dims_.size();
    }
    
    int Shape::operator[](std::size_t i) const
    {
	return dims_[i];
    }

    bool Shape::defined() const
    {
	for (auto x : dims_)
	    if (x == -1)
		return false;
	return true;
    }
    
    int Shape::total() const
    {
	int res = 1;
	for (auto x : dims_)
	    res *= x;
	return res;
    }

    bool operator==(const Shape& a, const Shape& b)
    {
	return a.dims() == b.dims();
    }
    
    bool operator!=(const Shape& a, const Shape& b)
    {
	return a.dims() != b.dims();
    }

    std::ostream& operator<<(std::ostream& os, const Shape& s)
    {
	os << "(";
	for (std::size_t i = 0; i < s.ndims(); ++i)
	{
	    os << s[i];
	    if (i + 1 < s.ndims())
		os << ", ";
	}
	return os << ")";
    }
    
}
