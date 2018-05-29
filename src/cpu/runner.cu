#include "runner.hh"
#include <iostream>
#include "../runtime/node.hh"
#include "../runtime/nodes-list.hh"
#include "kernels.hh"

namespace cpu
{

    void run_sequential(rt::NodesList& tasks)
    {
        for (auto x : tasks.nodes())
            kernels_list[x->type](x);
    }

}
