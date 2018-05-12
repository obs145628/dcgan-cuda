#include "adam-optimizer.hh"
#include "../ops/graph.hh"
#include "../ops/ops-builder.hh"
#include "../ops/seq.hh"
#include "../ops/update.hh"
#include "../ops/variable.hh"

#include <iostream>


AdamOptimizer::AdamOptimizer(dbl_t learning_rate,
                             dbl_t beta1,
                             dbl_t beta2,
                             dbl_t epsilon)
    : lr_(learning_rate)
    , beta1_(beta1)
    , beta2_(beta2)
    , eps_(epsilon)
{}

ops::Op* AdamOptimizer::minimize(ops::Op* loss)
{
    auto& graph = ops::Graph::instance();
    auto& builder = ops::OpsBuilder::instance();
    auto vars = graph.train_vars_get(loss);

    auto coeff = builder.variable(ops::Shape(), false);
    coeff->extend_name("sgd_coeff");
    *(coeff->data_begin()) = - lr_;

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
