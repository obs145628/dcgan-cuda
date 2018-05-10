#pragma once

#include "initializer.hh"

class CopyInitializer : public Initializer
{
public:

    CopyInitializer(const dbl_t* data);
    
    void fill(dbl_t* begin, dbl_t* end) override;

private:
    const dbl_t* data_;
};
