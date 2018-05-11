#include "update.hh"
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

    Update::Update(Variable* var, Op* dt, Op* coeff)
        : Op("update", var->shape_get(), {var, dt, coeff})
        , var_(var)
    {}

    void Update::compile()
    {

        auto& g = Graph::instance();

        auto& cdt = g.compiled(preds()[1]);
        auto& ccoeff = g.compiled(preds()[2]);

        Shape out_shape  = cdt.out_shape;
        std::size_t len = out_shape.total();
        dbl_t* ptr = var_->data_begin();

        auto out_node = rt::Node::op_update(ptr, cdt.out_data, ccoeff.out_data,
                                            len,
                                         {cdt.out_node, ccoeff.out_node});
        g.add_compiled(this, {out_node}, {}, out_node, out_shape, ptr);
        
    }

}
