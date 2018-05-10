#include "normal-initializer.hh"

NormalInitializer::NormalInitializer(dbl_t mean, dbl_t sd)
    : dist_(mean, sd)
{}

void NormalInitializer::fill(dbl_t* begin, dbl_t* end)
{
    while (begin != end)
        *begin++ = dist_(engine_) * 1.01;
}
