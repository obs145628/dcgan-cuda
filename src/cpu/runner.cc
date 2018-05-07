#include "runner.hh"
#include <iostream>
#include "../runtime/node.hh"
#include "kernels.hh"

namespace cpu
{

    void run_sequential(std::vector<rt::Node*> tasks)
    {
        for (auto x : tasks)
            kernels_list[x->type](x);
    }

}
