#pragma once

#include <atomic>
#include <map>
#include <vector>
#include "../runtime/fwd.hh"

namespace cpu
{

    struct RuntimeInfos
    {

        /**
         * When true, the thread pool is being deleted
         * All threads must stop
         */
        bool quit;

        //True if threads must wait for new tasks to be executed
        bool exec_graph;
        //True when the graph finished to be computed
        bool graph_completed;
        //exec_graph pass to false when all tasks have an asigned thread,
        //but graph_completed is set to true when all those are finished

        const std::vector<rt::Node*>* tasks_;
        std::atomic<std::size_t> next_task_;

        //map from node to indexes in tasks vector
        std::map<rt::Node*, std::size_t> indexes_;

        //1 is finished, 0 if not
        std::vector<int> tasks_status_;
    };
    
}
