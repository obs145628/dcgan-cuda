#pragma once

#include "copy.hh"
#include <algorithm>

inline void tensor_fill(dbl_t* begin, dbl_t* end, dbl_t val)
{
    while (begin != end)
	*begin++ = val;
}

inline void tensor_write(dbl_t* obegin, dbl_t* oend, const dbl_t* ibegin)
{
    std::copy(ibegin, ibegin + (oend - obegin), obegin);
}


inline void tensor_read(const dbl_t* ibegin, const dbl_t* iend, dbl_t* obegin)
{
    std::copy(ibegin, iend, obegin);
}
