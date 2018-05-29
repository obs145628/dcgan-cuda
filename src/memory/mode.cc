#include "mode.hh"
#include <cstdlib>
#include <cstring>
#include "../cpu/kernels.hh"

namespace
{
    ProgramMode compute_mode()
    {
        auto mode = getenv("RT_MODE");
        if (mode == nullptr)
            return ProgramMode::MONOTHREAD;
        else if (!strcmp(mode, "CPU"))
            return ProgramMode::MONOTHREAD;
        else if (!strcmp(mode, "MCPU"))
        {
            cpu::kernels_init();
            return ProgramMode::MULTITHREAD;
        }
        else if (!strcmp(mode, "GPU"))
            return ProgramMode::GPU;
        else
            return ProgramMode::MONOTHREAD;
    }
}


ProgramMode program_mode()
{
    static ProgramMode res = ProgramMode::UNDEFINED;

    if (res == ProgramMode::UNDEFINED)
        res = compute_mode();
    
    return res;
}
