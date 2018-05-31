#include "zero-initializer.hh"
#include "../memory/mode.hh"
#include "../memory/copy.hh"

void ZeroInitializer::fill(dbl_t* begin, dbl_t* end)
{

    dbl_t* tbegin = begin;
    dbl_t* tend = end;

    if (program_mode() == ProgramMode::GPU)
    {
        begin = new dbl_t[tend - tbegin];
        end = begin + (tend - tbegin);
    }
    

    for (auto it = begin; it != end; ++it)
        *it = 0;

    if (program_mode() == ProgramMode::GPU)
    {
        tensor_write(tbegin, tend, begin);
        delete[] begin;
    }
}
