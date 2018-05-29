#include "kernels.hh"
#include "runner.hh"
#include "../runtime/node.hh"
#include "../runtime/nodes-list.hh"
#include <stdexcept>

namespace gpu
{

    void run(rt::NodesList& tasks)
    {

        for (auto x : tasks.nodes())
        {
            kernels_list[x->type](x);
            cudaDeviceSynchronize();
        }
    }
    
}
