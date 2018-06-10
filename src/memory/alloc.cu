#include "alloc.hh"
#include "mode.hh"

dbl_t* tensor_alloc(std::size_t size)
{
    if (program_mode() == ProgramMode::GPU)
    {
        dbl_t* res;
        cudaMalloc(&res, size * sizeof(dbl_t));
        return res;
    }
    else
        return new dbl_t[size];
}

void tensor_free(dbl_t* ptr)
{
    if (program_mode() == ProgramMode::GPU)
        cudaFree(ptr);
    else
        delete[] ptr;
}
