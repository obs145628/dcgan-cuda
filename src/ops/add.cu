#include "add.hh"
#include <cassert>
#include <stdexcept>
#include "graph.hh"
#include "ops-builder.hh"
#include "sigmoid-grad.hh"
#include "../runtime/node.hh"
#include "../memory/alloc.hh"

namespace ops
{

    Add::Add(Op* left, Op* right)
        : Op("add", left->shape_get(), {left, right})
    {}

    void Add::compile()
    {
        auto& g = Graph::instance();
        auto& cleft = g.compiled(preds()[0]);
        auto& cright = g.compiled(preds()[1]);
        
        std::size_t len = cleft.out_shape.total();
        Shape out_shape = cleft.out_shape;
        dbl_t* out_data = tensor_alloc(len);

        auto out_node = rt::Node::op_add(cleft.out_data, cright.out_data, out_data,
                                         len,
                                         {cleft.out_node, cright.out_node});

        g.add_compiled(this, {out_node}, {out_data}, out_node, out_shape, out_data);
    }


    Op* Add::child_grad(std::size_t index, Op* dout)
    {
        assert(index < 2);
        (void) index;
        return dout;
    }
}
