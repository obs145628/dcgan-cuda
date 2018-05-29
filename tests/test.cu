#include <iostream>
#include <cmath>

#include "../src/memory/mode.hh"

int main()
{
    
    auto mode = program_mode();
    if (mode == ProgramMode::UNDEFINED)
        std::cout << "undefined\n";
    else if (mode == ProgramMode::MONOTHREAD)
        std::cout << "monothread\n";
    else if (mode == ProgramMode::MULTITHREAD)
        std::cout << "multithread\n";
    else if (mode == ProgramMode::GPU)
        std::cout << "gpu\n";
    
}
