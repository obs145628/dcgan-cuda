#include <iostream>
#include <cmath>

#include "../src/utils/dot-graph.hh"

#include "../src/cpu/thread-pool-runner.hh"

int main()
{

    cpu::ThreadPoolRunner pool(4);
    (void) pool;

}
