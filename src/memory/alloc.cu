#include "alloc.hh"
#include "mode.hh"

#include <stdexcept>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
          throw std::runtime_error {"Exec it\n"};
   }
}

dbl_t* tensor_alloc(std::size_t size)
{
    if (program_mode() == ProgramMode::GPU)
    {   
        dbl_t* res;
        gpuErrchk(cudaMalloc(&res, size * sizeof(dbl_t)));   
        return res;
    }
    else
        return new dbl_t[size];
}

void tensor_free(dbl_t* ptr)
{
    if (program_mode() == ProgramMode::GPU)
    {
        gpuErrchk(cudaFree(ptr));
    }
    else
        delete[] ptr;

    
}
