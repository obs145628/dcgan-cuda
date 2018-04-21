#pragma once

#include "memory.hh"

inline void tensor_fill(dbl_t* begin, dbl_t* end, dbl_t val)
{
    while (begin != end)
	*begin++ = val;
}
