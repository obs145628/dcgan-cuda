#include "adam-optimizer.hh"
#include "../ops/graph.hh"
#include "../ops/adam-update.hh"
#include "../ops/moment-update.hh"
#include "../ops/ops-builder.hh"
#include "../ops/seq.hh"
#include "../ops/update.hh"
#include "../ops/variable.hh"
#include "../memory/copy.hh"

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

ops::Op* AdamOptimizer::minimize(ops::Op* loss, const std::vector<ops::Variable*>& pvars)
{
    auto& graph = ops::Graph::instance();
    auto& builder = ops::OpsBuilder::instance();
    auto vars = pvars;
    if (vars.empty())
        vars = graph.train_vars_get(loss);

    std::vector<ops::Op*> updates;

    for (auto v : vars)
    {
        auto v_grad = graph.gradient(loss, v);
        v_grad->extend_name("adam_grad");
        
        auto v_dm = builder.variable(v->shape_get(), false);
        v_dm->extend_name("adam_m");
        tensor_fill(v_dm->data_begin(), v_dm->data_end(), 0);

        auto v_dv = builder.variable(v->shape_get(), false);
        v_dv->extend_name("adam_v");
        tensor_fill(v_dv->data_begin(), v_dv->data_end(), 0);

        auto dm_update = builder.moment_update(v_dm, v_grad, beta1_, 1 - beta1_, false);
        auto dv_update = builder.moment_update(v_dv, v_grad, beta2_, 1 - beta2_, true);

        auto v_update = builder.adam_update(v, dm_update, dv_update,
                                            lr_, beta1_, beta2_, eps_);
        
        updates.push_back(v_update);
    }

    auto train_op = builder.seq(updates);
    train_op->extend_name("adam_train");
    return train_op;
    
}
