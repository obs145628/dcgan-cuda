#include "memory.hh"

dbl_t* tensor_alloc(std::size_t size)
{
    return new dbl_t[size];
}

void tensor_free(dbl_t* ptr)
{
    delete[] ptr;
}
