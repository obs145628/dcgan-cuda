#include <iostream>
#include <cmath>

#include "../src/utils/dot-graph.hh"

#include "../src/cpu/thread-pool-runner.hh"

#include "../src/memory/mode.hh"

int main()
{

    cpu::ThreadPoolRunner pool(4);
    (void) pool;

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
