#include "runner.hh"
#include <stdexcept>

namespace gpu
{

    void run(rt::NodesList&)
    {
        throw std::runtime_error {"GPU mode not implemented"};
    }
    
}
