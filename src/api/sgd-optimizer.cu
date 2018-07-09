#include "sgd-optimizer.hh"
#include "../ops/graph.hh"
#include "../ops/ops-builder.hh"
#include "../ops/seq.hh"
#include "../ops/update.hh"
#include "../ops/variable.hh"
#include "../memory/copy.hh"

#include <iostream>


SGDOptimizer::SGDOptimizer(dbl_t learning_rate)
    : learning_rate_(learning_rate)
{}

ops::Op* SGDOptimizer::minimize(ops::Op* loss,
                                const std::vector<ops::Variable*>& pvars)
{
    auto& graph = ops::Graph::instance();
    auto& builder = ops::OpsBuilder::instance();
    auto vars = pvars;
    if (vars.empty())
        vars = graph.train_vars_get(loss);

    auto coeff = builder.variable(ops::Shape(), false);
    coeff->extend_name("sgd_coeff");

    dbl_t val = - learning_rate_;
    tensor_write(coeff->data_begin(), coeff->data_end(), &val);

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
