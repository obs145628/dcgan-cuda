#pragma once

#include <vector>
#include "../runtime/fwd.hh"

namespace cpu
{

    void run_sequential(std::vector<rt::Node*> tasks);
    
}
