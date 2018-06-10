#include "moment-update.hh"
#include <cassert>
#include <stdexcept>
#include "graph.hh"
#include "mse-grad.hh"
#include "ops-builder.hh"
#include "variable.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    MomentUpdate::MomentUpdate(Variable* var, Op* dt,
                               dbl_t coeff1, dbl_t coeff2, bool sq_update)
        : Op("moment_update", var->shape_get(), {var, dt})
        , var_(var)
        , coeff1_(coeff1)
        , coeff2_(coeff2)
        , sq_update_(sq_update)
    {}

    void MomentUpdate::compile()
    {

        auto& g = Graph::instance();

        auto& cdt = g.compiled(preds()[1]);

        Shape out_shape  = cdt.out_shape;
        std::size_t len = out_shape.total();
        dbl_t* ptr = var_->data_begin();

        
        
        auto out_node = sq_update_?
            rt::Node::op_moment_update2(ptr, cdt.out_data,
                                        coeff1_, coeff2_, len,
                                        {cdt.out_node})
            : rt::Node::op_moment_update(ptr, cdt.out_data,
                                         coeff1_, coeff2_, len,
                                         {cdt.out_node});
        g.add_compiled(this, {out_node}, {}, out_node, out_shape, ptr);
    }

}
