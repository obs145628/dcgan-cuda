#pragma once

#include "copy.hh"
#include "mode.hh"
#include <algorithm>

inline void tensor_fill(dbl_t* begin, dbl_t* end, dbl_t val)
{

    dbl_t* tbegin = begin;
    dbl_t* tend = end;

    if (program_mode() == ProgramMode::GPU)
    {
        begin = new dbl_t[tend - tbegin];
        end = begin + (tend - tbegin);
    }

    for (auto it = begin; it != end; ++it)
        *it = val;

    if (program_mode() == ProgramMode::GPU)
    {
        tensor_write(tbegin, tend, begin);
        delete[] begin;
    }
}

inline void tensor_write(dbl_t* obegin, dbl_t* oend, const dbl_t* ibegin)
{
    if (program_mode() == ProgramMode::GPU)
        cudaMemcpy(obegin, ibegin, (oend - obegin) * sizeof(dbl_t),
                   cudaMemcpyHostToDevice);
    else
        std::copy(ibegin, ibegin + (oend - obegin), obegin);
}


inline void tensor_read(const dbl_t* ibegin, const dbl_t* iend, dbl_t* obegin)
{
    if (program_mode() == ProgramMode::GPU)
        cudaMemcpy(obegin, ibegin, (iend - ibegin) * sizeof(dbl_t),
                   cudaMemcpyDeviceToHost);
    else
        std::copy(ibegin, iend, obegin);
}
