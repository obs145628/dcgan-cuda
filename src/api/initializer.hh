#pragma once

#include "../memory/types.hh"

class Initializer
{
public:
    virtual void fill(dbl_t* begin, dbl_t* end) = 0;
};
