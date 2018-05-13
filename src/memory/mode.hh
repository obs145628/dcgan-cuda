#pragma once

enum class ProgramMode
{
    UNDEFINED,
    MONOTHREAD,
    MULTITHREAD,
    GPU
};

ProgramMode program_mode();
