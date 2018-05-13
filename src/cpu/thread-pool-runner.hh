#pragma once

#include <thread>
#include <vector>

#include "fwd.hh"
#include "../runtime/fwd.hh"

namespace cpu
{

    class ThreadPoolRunner
    {

    public:
        ThreadPoolRunner(std::size_t nthreads);
        ~ThreadPoolRunner();

        void run(rt::NodesList& tasks);
        
    private:
        std::vector<std::thread> ths_;
        RuntimeInfos* infos_;
    };
    
}
