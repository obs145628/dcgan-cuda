#include "normal-initializer.hh"
#include "../memory/mode.hh"
#include "../memory/copy.hh"
#include "../utils/date.hh"

NormalInitializer::NormalInitializer(dbl_t mean, dbl_t sd)
    : engine_(date::now())
    , dist_(mean, sd)
{}

void NormalInitializer::fill(dbl_t* begin, dbl_t* end)
{
 
    dbl_t* tbegin = begin;
    dbl_t* tend = end;

    if (program_mode() == ProgramMode::GPU)
    {
        begin = new dbl_t[tend - tbegin];
        end = begin + (tend - tbegin);
    }

    for (auto it = begin; it != end; ++it)
        *it = next();

    if (program_mode() == ProgramMode::GPU)
    {
        tensor_write(tbegin, tend, begin);
        delete[] begin;
    }
}

dbl_t NormalInitializer::next()
{
    return dist_(engine_);
}
