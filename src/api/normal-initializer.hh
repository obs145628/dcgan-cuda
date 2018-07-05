#pragma once

#include <random>
#include "initializer.hh"

class NormalInitializer : public Initializer
{
public:
    NormalInitializer(dbl_t mean = 0, dbl_t sd = 1);
    void fill(dbl_t* begin, dbl_t* end) override;
    dbl_t next();

private:
    std::mt19937 engine_;
    std::normal_distribution<dbl_t> dist_;
};
