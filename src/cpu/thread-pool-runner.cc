#include "thread-pool-runner.hh"
#include "runtime-infos.hh"
#include "kernels.hh"
#include "../runtime/node.hh"

#include <chrono>
#include <iostream>

namespace cpu
{


    namespace
    {

        bool task_ready(rt::Node* node, RuntimeInfos* infos)
        {
            for (auto n : node->preds)
                if (!infos->tasks_status_[infos->indexes_[n]])
                    return false;
            return true;
        }
        

        void exec_graph(RuntimeInfos* infos)
        {

            while (infos->exec_graph)
            {

                std::size_t task = infos->next_task_++;
                if (task >= infos->tasks_->size())
                    break;

                rt::Node* node = infos->tasks_->at(task);
                //wait for predecessors to finish
                while (!task_ready(node, infos))
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                kernels_list[node->type](node);
                infos->tasks_status_[infos->indexes_[node]] = 1;
            }

            //no more operands to be executed but some didn't finish executing operations
            infos->exec_graph = false;

            //set flag if graph terminated
            for (auto s : infos->tasks_status_)
                if (s == 0)
                    return;
            infos->graph_completed = true;
        }

        void thread_runner(RuntimeInfos* infos)
        {

            //std::cout << "begin\n";

            while (!infos->quit)
            {

                if (infos->exec_graph)
                    exec_graph(infos);
                else
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            //std::cout << "end\n";
            
        }
        
    }
    
    

    ThreadPoolRunner::ThreadPoolRunner(std::size_t nthreads)
    {

        infos_ = new RuntimeInfos;
        infos_->quit = false;
        infos_->exec_graph = false;

        for (std::size_t i = 0; i < nthreads; ++i)
            ths_.emplace_back(&thread_runner, infos_); 
    }

    ThreadPoolRunner::~ThreadPoolRunner()
    {
        infos_->quit = true;
        for (auto& t : ths_)
            t.join();
        delete infos_;
    }

    void ThreadPoolRunner::run(std::vector<rt::Node*>& tasks)
    {
        infos_->tasks_ = &tasks;
        infos_->next_task_ = 0;
        infos_->graph_completed = false;

        infos_->indexes_.clear();
        for (std::size_t i = 0; i < tasks.size(); ++i)
            infos_->indexes_[tasks[i]] = i;
        infos_->tasks_status_ = std::vector<int>(tasks.size(), 0);
        
        infos_->exec_graph = true;
        while (!infos_->graph_completed)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
    }
    
}
