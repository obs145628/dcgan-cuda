#pragma once

#include "initializer.hh"

class ZeroInitializer : public Initializer
{
public:
    void fill(dbl_t* begin, dbl_t* end) override;
};
