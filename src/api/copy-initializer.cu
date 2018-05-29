#include "copy-initializer.hh"
#include "../memory/copy.hh"

CopyInitializer::CopyInitializer(const dbl_t* data)
    : data_(data)
{}

void CopyInitializer::fill(dbl_t* begin, dbl_t* end)
{
    tensor_write(begin, end, data_);
}
