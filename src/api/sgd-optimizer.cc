#include "sgd-optimizer.hh"
#include "../ops/graph.hh"
#include "../ops/ops-builder.hh"
#include "../ops/seq.hh"
#include "../ops/update.hh"
#include "../ops/variable.hh"

#include <iostream>


SGDOptimizer::SGDOptimizer(dbl_t learning_rate)
    : learning_rate_(learning_rate)
{}

ops::Op* SGDOptimizer::minimize(ops::Op* loss)
{
    auto& graph = ops::Graph::instance();
    auto& builder = ops::OpsBuilder::instance();
    auto vars = graph.train_vars_get(loss);
    

    std::cout << vars.size() << std::endl;

    auto coeff = builder.variable(ops::Shape(), false);
    coeff->extend_name("sgd_coeff");
    *(coeff->data_begin()) = - learning_rate_;

    std::vector<ops::Op*> updates;

    for (auto v : vars)
    {
        auto v_grad = graph.gradient(loss, v);
        v_grad->extend_name("sgd_grad");
        updates.push_back(builder.update(v, v_grad, coeff));
    }

    auto train_op = builder.seq(updates);
    train_op->extend_name("sgd_train");
    return train_op;
    
}
