#include "adam-update.hh"
#include <cassert>
#include <stdexcept>
#include "graph.hh"
#include "mse-grad.hh"
#include "ops-builder.hh"
#include "variable.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"
#include "../memory/copy.hh"

namespace ops
{

    AdamUpdate::AdamUpdate(Variable* var, Op* m, Op* v,
                           dbl_t learning_rate,
                           dbl_t beta1, dbl_t beta2, dbl_t eps)
        : Op("adam_update", var->shape_get(), {var, m, v})
        , var_(var)
        , t_(0)
        , lr_(learning_rate)
        , beta1_(beta1)
        , beta2_(beta2)
        , eps_(eps)
    {}

    AdamUpdate::~AdamUpdate()
    {

    }

    void AdamUpdate::compile()
    {

        auto& g = Graph::instance();

        auto& cm = g.compiled(preds()[1]);
        auto& cv = g.compiled(preds()[2]);

        Shape out_shape  = cm.out_shape;
        std::size_t len = out_shape.total();
        dbl_t* ptr = var_->data_begin();
        
        
        auto out_node = rt::Node::op_adam_update(ptr, &t_, cm.out_data, cv.out_data,
                                                 lr_, beta1_, beta2_, eps_,
                                                 len,
                                                 {cm.out_node, cv.out_node});
        g.add_compiled(this, {out_node}, {}, out_node, out_shape, ptr);
    }

}
