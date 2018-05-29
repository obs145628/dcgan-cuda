#include "zero-initializer.hh"

void ZeroInitializer::fill(dbl_t* begin, dbl_t* end)
{
    while (begin != end)
        *begin++ = 0;
}
